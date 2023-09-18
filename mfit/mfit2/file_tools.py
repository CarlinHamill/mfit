"""
This file is part of Macromolecular Rate Theory (MMRT) Fitting Tool (MFIT)
Copyright (C), 2023, Carlin Hamill.

MFIT is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
import pandas as pd
import json
import typing
from lmfit.models import LinearModel

from mfit.mfit2.data_tools import Dataset, RateType, Inputs

class FileInput(list):
    def __init__(self, *args, **kwargs)->None:
        super(FileInput, self).__init__(*args, **kwargs)

    @staticmethod
    def convert_from_deltag(name:str, temperature:np.ndarray, deltag_input:np.ndarray, raw_array:np.ndarray)->Dataset:
        lnrate = 23.76014 + np.log(temperature) - (deltag_input/(8.314 * temperature))
        rate = np.exp(lnrate)

        inputs = Inputs(temperature, rate, deltag_input, lnrate, RateType.DeltaG, raw_array)
        return Dataset(name, inputs)

    @staticmethod
    def convert_from_lnrate(name:str, temperature:np.ndarray, lnrate_input:np.ndarray, raw_array:np.ndarray)->Dataset:
        deltag = 8.314 * temperature * (23.76014 + np.log(temperature) - lnrate_input)
        rate = np.exp(lnrate_input)
        
        inputs = Inputs(temperature, rate, deltag, lnrate_input, RateType.lnRate, raw_array)
        return Dataset(name, inputs)

    @staticmethod
    def convert_from_rate(name:str, temperature:np.ndarray, rate_input:np.ndarray, raw_array:np.ndarray)->Dataset:
        lnrate = np.log(rate_input)
        deltag = 8.314 * temperature * (23.76014 + np.log(temperature) - lnrate)

        inputs = Inputs(temperature, rate_input, deltag, lnrate, RateType.Rate, raw_array)
        return Dataset(name, inputs)
    
    def find_conversion_function(self, temperature:np.ndarray, rate: np.ndarray)->typing.Callable:
        # Find gradient of first four datapoints
        restricted_temperature = temperature[:4]
        restricted_rate = rate[:4]

        model = LinearModel()
        pars = model.guess(restricted_rate, x=restricted_temperature)
        out = model.fit(restricted_rate, pars, x=restricted_temperature)
        slope = out.values['slope']
        
        if slope <= 0:
            return self.convert_from_deltag

        # Find gradients of datapoints 1 -> Max
        max_idx = np.argmax(rate)
        temperature = temperature[:max_idx]
        rate = rate[:max_idx]
        
        slopes = [self.get_slope(temperature[li:ui],rate[li:ui]) for (li, ui) in self.get_elements(len(temperature))]

        difference = max(slopes) - min(slopes)

        if difference <= 0.5:
            return self.convert_from_lnrate
        return self.convert_from_rate

    @staticmethod
    def get_slope(temperature:np.ndarray, rate:np.ndarray)->float:
        model = LinearModel()
        pars = model.guess(rate, x=temperature)
        out = model.fit(rate, pars, x=temperature)
        return out.values['slope']
    
    @staticmethod
    def get_elements(n:int)->tuple[int]:
        for i in range(n):
            for j in range(n):
                if i + 2 <= j:
                    yield(i,j)

    @staticmethod
    def is_string(val):
        try:
            float(val)
        except ValueError:
            return True
        return False

    def parse_raw_input(self, name:str, sheet:pd.DataFrame)->Dataset:
        #remove heading row if present
        if any([self.is_string(val) for val in list(sheet[:1].values[0])]):
            sheet = sheet.iloc[1:,:]
        
        # Read temperature array
        temperature_array = sheet.pop(sheet.columns[0])
        temperature_array = temperature_array.apply(pd.to_numeric, errors='coerce')
        temperature_array = np.array(temperature_array, dtype=np.float64)
        temperature_array = temperature_array[~np.isnan(temperature_array)]

        if temperature_array[0] < 150:
            temperature_array += 273.15

        # Iterate over rows, convert to numeric, average rows, then combine into single numpy array and remove NaN values
        raw_rate_array = np.array([row[1].apply(pd.to_numeric, errors='coerce') for row in sheet.iterrows()])
        raw_rate_array = raw_rate_array[~np.isnan(raw_rate_array).all(axis=1)]
        average_rate_array = np.nanmean(raw_rate_array, axis=1)
        average_rate_array = average_rate_array[~np.isnan(average_rate_array)]

        # Sort arrays from low -> high temperature
        sort_index = np.argsort(temperature_array)
        temperature_array = temperature_array[sort_index]
        average_rate_array = average_rate_array[sort_index]

        # Generate all deltag/lnrate/rate arrays and create dataset object
        conversion_function = self.find_conversion_function(temperature_array, average_rate_array)
        dataset = conversion_function(name, temperature_array, average_rate_array, raw_rate_array)

        return dataset


class ExcelInput(FileInput):
    def __init__(self, file_name:str, *args, **kwargs)->None:
        super(ExcelInput, self).__init__(*args, **kwargs)
        self.file_name = file_name 
        self.import_excel()

    def import_excel(self)->None:
        with pd.ExcelFile(self.file_name) as excel_input:
            data = {sheet_name: excel_input.parse(sheet_name, header=None) for sheet_name in excel_input.sheet_names}
        for name, sheet in data.items():
            self.append(self.parse_raw_input(name, sheet))


class CSVInput(FileInput):
    def __init__(self, file_name:str, *args, **kwargs)->None:
        super(CSVInput, self).__init__(*args, **kwargs)
        self.file_name = file_name
        self.import_csv()

    def import_csv(self)->None:
        csv = pd.read_csv(self.file_name, sep=r"\,\s*", engine='python', header=None)
        name = self.file_name.split('.')[-2].split('/')[-1]
        self.append(self.parse_raw_input(name, csv))


class TabDelimitedInput(FileInput):
    def __init__(self, file_name:str, *args, **kwargs)->None:
        super(TabDelimitedInput, self).__init__(*args, **kwargs)
        self.file_name = file_name
        self.import_tabs()

    def import_tabs(self):    
        csv = pd.read_csv(self.file_name, sep='\t', engine='python', header=None)
        name = self.file_name.split('.')[-2].split('/')[-1]
        self.append(self.parse_raw_input(name, csv))


class SpaceDelimitedInput(FileInput):
    def __init__(self, file_name:str, *args, **kwargs)->None:
        super(SpaceDelimitedInput, self).__init__(*args, **kwargs)
        self.file_name = file_name
        self.import_space_delimited()

    def import_space_delimited(self)->None:
        csv = pd.read_csv(self.file_name, sep='\s+', engine='python', header=None)
        name = self.file_name.split('.')[-2].split('/')[-1]
        self.append(self.parse_raw_input(name, csv))


class MMRTFileInput(FileInput):
    def __init__(self, file_name:str, *args, **kwargs)->None:
        super(MMRTFileInput, self).__init__(*args, **kwargs)
        self.file_name = file_name
        self.import_mmrt_file()

    def import_mmrt_file(self)->None:
        with open(self.file_name, 'r') as infile:
            datasets = json.load(infile)

        for dataset in datasets.values():
            self.append(
                Dataset.from_json(dataset)
            )
