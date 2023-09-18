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

from typing import TYPE_CHECKING, ValuesView, ItemsView, Union
import typing
if TYPE_CHECKING:
    import matplotlib.pyplot as plt

from dataclasses import dataclass, field
import json
from enum import Enum

import sys
import threading
import _thread as thread
 
#Module Imports
import numpy as np
from lmfit.models import ExpressionModel
import lmfit as lm
import scipy
import numba


class RateType(Enum):    
    Rate = 0
    lnRate = 1
    DeltaG = 2


class ParamVariableType(Enum):
    Free = 0
    Fixed = 1
    Auto = 2


@dataclass
class Inputs:
    temperature: np.ndarray
    rate: np.ndarray
    deltag: np.ndarray
    lnrate: np.ndarray
    rate_type: RateType
    raw: np.ndarray = None

    def get_input_rate(self)->tuple[np.ndarray]:
        if self.rate_type == RateType.DeltaG:
            return self.temperature, self.deltag, self.raw
        if self.rate_type == RateType.lnRate:
            return self.temperature, self.lnrate, self.raw
        return self.temperature, self.rate, self.raw

    @property
    def deltag_all(self)->np.ndarray:
        if self.rate_type == RateType.DeltaG:
            return self.raw
        if self.rate_type == RateType.lnRate:
            dg = 8.314 * self.temperature[:,None] * (23.76014 + np.log(self.temperature[:,None]) - self.raw)
            return dg
        if self.rate_type == RateType.Rate:
            dg = 8.314 * self.temperature[:,None] * (23.76014 + np.log(self.temperature[:,None]) - np.log(self.raw))
            return dg

    @property
    def deltag_error(self)->np.ndarray:
        t = self.temperature[:,None]
        if self.rate_type == RateType.DeltaG:
            return np.nanstd(self.raw, axis=1)
        if self.rate_type == RateType.lnRate:
            dg = 8.314 * t * (23.76014 + np.log(t) - self.raw)
            return np.nanstd(dg, axis=1)
        dg = 8.314 * t * (23.76014 + np.log(t) - np.log(self.raw))
        return np.nanstd(dg, axis=1)

    @classmethod
    def from_deltag(cls, temperature:np.ndarray, deltag:np.ndarray, raw_array:np.ndarray):#->Self:
        lnrate = 23.76014 + np.log(temperature) - (deltag/(8.314 * temperature))
        rate = np.exp(lnrate)
        return cls(temperature, rate, deltag, lnrate, RateType.DeltaG, raw_array)
    @classmethod
    def from_lnrate(cls, temperature:np.ndarray, lnrate:np.ndarray, raw_array:np.ndarray):#->Self:
        deltag = 8.314 * temperature * (23.76014 + np.log(temperature) - lnrate)
        rate = np.exp(lnrate)
        return cls(temperature, rate, deltag, lnrate, RateType.lnRate, raw_array)
    @classmethod
    def from_rate(cls, temperature:np.ndarray, rate:np.ndarray, raw_array:np.ndarray):#->Self:
        lnrate = np.log(rate)
        deltag = 8.314 * temperature * (23.76014 + np.log(temperature) - lnrate)
        return cls(temperature, rate, deltag, lnrate, RateType.Rate, raw_array)
    
    @classmethod
    def from_json(cls, inputs:dict[str, list[float]]):#->Self:
        return cls(
            np.array(inputs['temperature']),
            np.array(inputs['rate']),
            np.array(inputs['deltag']),
            np.array(inputs['lnrate']),
            RateType(inputs['rate_type']),
            np.array(inputs['raw'])
        )


@dataclass
class Parameter:
    value: Union[int, float]
    variable: ParamVariableType = None

    def to_json(self)->tuple[Union[float, int]]:
        if self.variable is not None:
            return (self.value, self.variable.value)
        return (self.value,)


@dataclass
class MMRT2_Parameters:
    H: Parameter = field(default_factory=lambda:Parameter(45000, ParamVariableType.Free))
    S: Parameter = field(default_factory=lambda:Parameter(100, ParamVariableType.Free))
    ddH: Parameter = field(default_factory=lambda:Parameter(90000, ParamVariableType.Free))
    CpH: Parameter = field(default_factory=lambda:Parameter(20000, ParamVariableType.Free))
    Tn: Parameter = field(default_factory=lambda:Parameter(278.15, ParamVariableType.Fixed))
    Tc: Parameter = field(default_factory=lambda:Parameter(None, ParamVariableType.Auto))
    CpL: Parameter = field(default_factory=lambda:Parameter(None, ParamVariableType.Auto))
    CpL_Point1: Parameter = field(default_factory=lambda:Parameter(1))
    CpL_Point2: Parameter = field(default_factory=lambda:Parameter(6))
    CpL_H: Parameter = field(default_factory=lambda: Parameter(45000))
    CpL_S: Parameter = field(default_factory=lambda: Parameter(100))
    CpL_CpL: Parameter = field(default_factory=lambda: Parameter(500))

    @classmethod
    def from_json(cls, parameters_dict:dict[list[Union[float, int]]]):
        if parameters_dict is None:
            return cls()
        return cls(
            H = Parameter(parameters_dict['H'][0], ParamVariableType(parameters_dict['H'][1])),
            S = Parameter(parameters_dict['S'][0], ParamVariableType(parameters_dict['S'][1])),
            ddH = Parameter(parameters_dict['ddH'][0], ParamVariableType(parameters_dict['ddH'][1])),
            CpH = Parameter(parameters_dict['CpH'][0], ParamVariableType(parameters_dict['CpH'][1])),
            Tn = Parameter(parameters_dict['Tn'][0], ParamVariableType(parameters_dict['Tn'][1])),
            Tc = Parameter(parameters_dict['Tc'][0], ParamVariableType(parameters_dict['Tc'][1])),
            CpL = Parameter(parameters_dict['CpL'][0], ParamVariableType(parameters_dict['CpL'][1])),
            CpL_Point1 = Parameter(parameters_dict['CpL_Point1'][0]),
            CpL_Point2 = Parameter(parameters_dict['CpL_Point2'][0]),
        )
        

class MMRTFitFailedError(Exception):
    def __init__(self,  *args, **kwargs) -> None:
        super(MMRTFitFailedError, self).__init__(*args, **kwargs)


class Fit_CpL:
    def __init__(self, inputs:Inputs, parameters:MMRT2_Parameters, mask:np.ndarray)->None:
        self.inputs = inputs
        self.parameters = parameters
        self.mask = mask

        try:
            self.fit_CpL()
        except Exception:
            # import traceback
            # print('\n\n'+'-'*120)
            # print(traceback.format_exc())
            # print('-'*120+'\n\n')
            self.CpL = None

    def fit_CpL(self)->None:
        cpl_equation = ExpressionModel("H - S * x + CpL * (x - 288 - x * log(x/288))") 
        
        start = self.parameters.CpL_Point1.value - 1
        end = self.parameters.CpL_Point2.value

        if (end-start) > len(self.inputs.temperature[self.mask!=0]):
            self.CpL = None
            return

        temp = self.inputs.temperature[self.mask!=0][start:end]
        deltag = self.inputs.deltag[self.mask!=0][start:end]
        
        self.CpL_fit = cpl_equation.fit(
            deltag, x=temp,
            calc_covar=True, 
            H=self.parameters.CpL_H.value, 
            S=self.parameters.CpL_S.value, 
            CpL=self.parameters.CpL_CpL.value
        )
        self.values = self.CpL_fit.values
        self.CpL = self.values['CpL']
        self.CpL_Error = self.CpL_fit.params['CpL'].stderr
        self.calculate_CI()

    def calculate_CI(self)->None:
        """
        method from https://kitchingroup.cheme.cmu.edu/blog/2013/02/12/\
            Nonlinear-curve-fitting-with-parameter-confidence-intervals/#\
            :~:text=A%20confidence%20interval%20tells%20us,which%20best%20fit%20the%20data.
        """
        try:
            alpha = 0.05
            n = len(self.inputs.temperature)
            p = 3
            dof = max(0, n - p)
            tval = scipy.stats.distributions.t.ppf(1.0-alpha/2., dof)
            var = np.diag(self.CpL_fit.covar)[2]
            self.ci = (self.CpL-(var**0.5)*tval, self.CpL+(var**0.5)*tval)
        except ValueError:
            self.ci = None

    def plot(self, ax:'plt.Axes', picker=None)->None:
        plot_x, plot_y = self.calculate_fit_curve()
        ax.plot(plot_x, plot_y, 'r-', label='_o1')
        start = self.parameters.CpL_Point1.value - 1
        end = self.parameters.CpL_Point2.value
        ax.plot(
            self.inputs.temperature[self.mask!=0][start:end], 
            self.inputs.deltag[self.mask!=0][start:end], 
            'r.', 
            label='_o1_points',
            zorder=5
        )

    def calculate_fit_curve(self, plot_density:int=1000, buffer:int=10)->tuple[np.ndarray]:
        min_temp = self.inputs.temperature[self.mask!=0][self.parameters.CpL_Point1.value -1]-buffer
        max_temp = self.inputs.temperature[self.mask!=0][self.parameters.CpL_Point2.value -1]+buffer
        x = np.linspace(min_temp, max_temp, plot_density)
        
        dg = (self.values['H']-self.values['S']*x+self.values['CpL']*(x-288-x*np.log(x/288)))
        
        return x, dg
      

def quit_function(fn_name:typing.Callable)->None:
    sys.stderr.flush() # Python 3 stderr is likely buffered.
    thread.interrupt_main() # raises KeyboardInterrupt


def exit_after(s:int)->None:
    '''
    use as decorator to exit process if 
    function takes longer than s seconds
    '''
    def outer(fn:typing.Callable):
        def inner(*args, **kwargs):
            timer = threading.Timer(s, quit_function, args=[fn.__name__])
            timer.start()
            try:
                result = fn(*args, **kwargs)
            finally:
                timer.cancel()
            return result
        return inner
    return outer


@numba.njit
def enthalpic_heat_capacity(t:float, cl:float, ch:float, ddh:float, tc:float)->float:
    cp = (cl + ch * np.exp((-ddh*(1-(t/tc)))/(8.314*t)))/(1 + np.exp((-ddh*(1-(t/tc)))/(8.314*t)))
    return cp


@numba.njit
def entropic_heat_capacity(t:float, cl:float, ch:float, ddh:float, tc:float)->float:
    cp = (cl + ch * np.exp((-ddh*(1-(t/tc)))/(8.314*t)))/(1 + np.exp((-ddh*(1-(t/tc)))/(8.314*t)))
    return cp/t


class FitMMRT2:
    def __init__(self, name:str, inputs:Inputs, parameters:MMRT2_Parameters, mask:np.ndarray)->None:
        self.name = name
        self.inputs = inputs
        self.parameters = parameters
        self.mask = mask
        
        self.fit_mmrt2()

    @property
    def r_squared(self)->tuple[str]|None:
        if self.mmrt2_fit is None:
            return (0,0)
        rsq = 1 - self.mmrt2_fit.residual.var()/ self.inputs.deltag_all.var()
        n = self.mmrt2_fit.ndata
        p = self.mmrt2_fit.nvarys
        arsq = rsq - (1-rsq) * p / (n-p-1)
        return (rsq, arsq)
    
    @property
    def aicc(self)->float:
        if self.mmrt2_fit is None:
            return 0
        aic = self.mmrt2_fit.aic
        aicc = aic + (2*(self.mmrt2_fit.nvarys**2)+2*self.mmrt2_fit.nvarys)/(self.mmrt2_fit.ndata-self.mmrt2_fit.nvarys-1)
        return aicc

    @property
    def fit_statistics(self)->dict[str, float]:
        statistics = {
            'ndata': self.mmrt2_fit.ndata if self.mmrt2_fit is not None else 0,
            'nvarys': self.mmrt2_fit.nvarys if self.mmrt2_fit is not None else 0,
            'rsq': self.r_squared[0],
            'arsq': self.r_squared[1],
            'aic': self.mmrt2_fit.aic if self.mmrt2_fit is not None else 0,
            'aicc': self.aicc,
            'chi_sq': self.mmrt2_fit.chisqr if self.mmrt2_fit is not None else 0,
            'chi_sq_reduced': self.mmrt2_fit.redchi if self.mmrt2_fit is not None else 0,
            'bic': self.mmrt2_fit.bic if self.mmrt2_fit is not None else 0
        }
        return statistics
    
    def fit_CpL(self)->None:
        if self.parameters.CpL.variable.value == 2:#Auto
            self.cpl_fit = Fit_CpL(self.inputs, self.parameters, self.mask)
            if self.cpl_fit.CpL is not None:
                self.parameters.CpL.value = self.cpl_fit.CpL 
            else:
                self.parameters.CpL.value = None
                raise MMRTFitFailedError
            return
        self.cpl_fit = None
    
    def find_tc(self)->None:
        if self.parameters.Tc.variable.value ==2:
            self.parameters.Tc.value = self.inputs.temperature[self.inputs.deltag.argmin()]       

    @staticmethod
    def rate(temperature_array:np.ndarray, params:dict|lm.Parameters)->np.ndarray:
        
        if isinstance(params, lm.Parameters):
            params = params.valuesdict()

        h = params['H']
        s = params['S']
        ch = params['CpH']
        cl = params['CpL']
        ddh = params['ddH']
        tc = params['Tc']
        tn = params['Tn']

        def heat_capacity_integral(temperature_array:np.ndarray, integral_function:typing.Callable)->np.ndarray:
            return np.array([scipy.integrate.quad(integral_function, tn, t, args=(cl, ch, ddh, tc))[0] for t in temperature_array])

        dg = h - temperature_array*s + heat_capacity_integral(temperature_array, enthalpic_heat_capacity) \
            - temperature_array*heat_capacity_integral(temperature_array, entropic_heat_capacity)
        return dg
    
    def deltag_residual(self, parameters:lm.Parameters, temperature_array:np.ndarray, deltag_array:np.ndarray)->np.ndarray:
        return self.rate(temperature_array, parameters) - deltag_array  
    
    def fit_mmrt2(self)->None:
        try:
            self.fit_CpL()
            self.find_tc()        
            self.run_fit()

        except (KeyboardInterrupt, Exception) as e:
            # import traceback
            # print('\n\n'+'-'*120)
            # print(traceback.format_exc())
            # print('-'*120+'\n\n')
            self.mmrt2_fit = None    

    def initialise_parameters(self)->lm.Parameters:
        pars = lm.Parameters()
        pars.add('H', value=self.parameters.H.value, vary=self.convert_param_variable_to_bool(self.parameters.H))
        pars.add('S', value=self.parameters.S.value, vary=self.convert_param_variable_to_bool(self.parameters.S))
        pars.add('CpH', value=self.parameters.CpH.value, vary=self.convert_param_variable_to_bool(self.parameters.CpH))
        pars.add('ddH', value=self.parameters.ddH.value, vary=self.convert_param_variable_to_bool(self.parameters.ddH))
        pars.add('Tn', value=self.parameters.Tn.value, vary=False)
        pars.add('CpL', value=self.parameters.CpL.value, vary=self.convert_param_variable_to_bool(self.parameters.CpL))
        pars.add('Tc', value=self.parameters.Tc.value, vary=self.convert_param_variable_to_bool(self.parameters.Tc), min=min(self.inputs.temperature), max=max(self.inputs.temperature))
        return pars

    def convert_param_variable_to_bool(self, param:ParamVariableType)->bool:
        if param.variable.value == 0:#Free
            return True
        if param.variable.value == 1:#Fixed
            return False
        if param.variable.value == 2:#Auto
            if param is self.parameters.Tc:
                return True
            return False


    @exit_after(5)
    def run_fit(self)->None:
        pars = self.initialise_parameters()
        self.minimizer = lm.Minimizer(
            self.deltag_residual, 
            pars, 
            fcn_args=(self.inputs.temperature[self.mask!=0], self.inputs.deltag[self.mask!=0]),
            calc_covar=True
        )
        self.mmrt2_fit = self.minimizer.minimize(method='leastsq')
        
        self.values = self.mmrt2_fit.params.valuesdict()
        self.error = {
            'H': self.mmrt2_fit.params['H'].stderr,
            'S': self.mmrt2_fit.params['S'].stderr,
            'CpH':self.mmrt2_fit.params['CpH'].stderr,
            'CpL': self.cpl_fit.CpL_Error if self.cpl_fit is not None else self.mmrt2_fit.params['CpL'].stderr,
            'ddH': self.mmrt2_fit.params['ddH'].stderr,
            'Tc': self.mmrt2_fit.params['Tc'].stderr,
            'Tn':self.mmrt2_fit.params['Tn'].stderr
        }
        self.calculate_CI()
        
    def calculate_CI(self)->None:
        """
        method from https://kitchingroup.cheme.cmu.edu/blog/2013/02/12/\
            Nonlinear-curve-fitting-with-parameter-confidence-intervals/#\
            :~:text=A%20confidence%20interval%20tells%20us,which%20best%20fit%20the%20data.
        """
        try:
            alpha = 0.05
            n = len(self.inputs.temperature)
            nfix = len([par for par in self.mmrt2_fit.params if self.mmrt2_fit.params[par].vary == False])
            p = len(self.mmrt2_fit.params) - nfix
            dof = max(0, n - p)
            tval = scipy.stats.distributions.t.ppf(1.0-alpha/2., dof)
            keys = [par_name for par_name in self.mmrt2_fit.params.valuesdict().keys() if self.mmrt2_fit.params[par_name].vary == True]
            vals = [val for (par_name, val) in self.mmrt2_fit.params.valuesdict().items() if self.mmrt2_fit.params[par_name].vary == True]
            self.ci = {n: (p - (var**0.5)*tval, p + (var**0.5)*tval) for (n, p, var) in zip(keys, vals, np.diag(self.mmrt2_fit.covar))}
        
            if self.cpl_fit is not None:
                self.ci.update({'CpL':self.cpl_fit.ci})
            else:
                self.ci.update({'CpL':0})

            ls =  ['H', 'S', 'CpL', 'CpH', 'ddH', 'Tc', 'Tn']
            self.ci.update({key:'None' for key in ls if key not in self.ci.keys()})

        except ValueError:
            self.ci = {}
            ls =  ['H', 'S', 'CpL', 'CpH', 'ddH', 'Tc', 'Tn']
            self.ci.update({key:'None' for key in ls if key not in self.ci.keys()})

    def plot(self, ax:'plt.Axes', picker:int=None)->None:
        temperature_data = self.inputs.temperature
        deltag_data = self.inputs.deltag
        temperature_curve, deltag_curve = self.calculate_fit_curve()

        ax.errorbar(
            x=temperature_data[self.mask!=0],
            y=deltag_data[self.mask!=0],
            yerr=self.inputs.deltag_error[self.mask!=0],
            fmt='o',
            capsize=2,
            label=self.get_label(),
            picker=picker if picker is not None else 5
        )

        colour = ax.get_lines()[-1].get_color()
        ax.plot(temperature_curve, deltag_curve, '-', label='_curve', color=colour)
        
        ax.errorbar(
            x=temperature_data[self.mask!=1],
            y=deltag_data[self.mask!=1],
            yerr=self.inputs.deltag_error[self.mask!=1],
            fmt='ro',
            capsize=2,
            label='_removed',
            picker=6
        )

        ax.legend(loc='best')
        ax.set_xlabel('Temperature (K)', fontsize=12)
        ax.set_ylabel(r'$\Delta$G$^\ddag$', fontsize=12)
    
    def calculate_fit_curve(self, plot_density:int=1000)->tuple[np.ndarray]:
        x = np.linspace(min(self.inputs.temperature), max(self.inputs.temperature), plot_density)
        return x, self.rate(x, self.values)
    
    def get_label(self)->str:
        if self.name != '':
            return self.name
        return ' '  


@dataclass
class FitMMRT:
    mmrt2: FitMMRT2 = None

    def fit_mmrt(self, name:str, inputs:Inputs, mmrt_parameters:MMRT2_Parameters, mask:np.ndarray)->None:
        self.mmrt2 = FitMMRT2(name, inputs, mmrt_parameters, mask)
        self.mmrt2_inputs = inputs

    def get_fit_curve(self)->tuple[np.ndarray]:
        if self.mmrt2.mmrt2_fit is not None:
            return self.mmrt2.calculate_fit_curve(plot_density=250)  
        else: 
            return (
                np.linspace(
                    min(self.mmrt2_inputs.temperature), 
                    max(self.mmrt2_inputs.temperature), 
                    250
                ), 
                np.zeros(250)
            )


@dataclass
class Dataset:
    name:str
    inputs: Inputs
    parameters: MMRT2_Parameters = field(default_factory=MMRT2_Parameters)
    fit: FitMMRT = field(default_factory=FitMMRT)
    mask:np.ndarray = None

    def fit_mmrt(self)->None:
        if self.mask is None:
            self.init_datapoint_mask()

        self.fit.fit_mmrt(
            self.name,
            self.inputs,
            self.parameters,
            self.mask
        )

    def change_input_type(self, rate_type:int)->None:
        temperature, rate_data, raw_array = self.inputs.get_input_rate()

        if rate_type == 0:
            self.inputs = Inputs.from_rate(temperature, rate_data, raw_array)
        if rate_type == 1:
            self.inputs = Inputs.from_lnrate(temperature, rate_data, raw_array)
        if rate_type == 2:
            self.inputs = Inputs.from_deltag(temperature, rate_data, raw_array)
        self.parameters.CpL.value = None
        self.parameters.Tc.value = None

    def change_name(self, new_name:str)->None:
        self.name = new_name
        if self.fit is not None:
            self.fit.name = new_name

    def plot(self, ax:'plt.Axes', picker=None)->None:
        ax.errorbar(
            x=self.inputs.temperature[self.mask!=0],
            y=self.inputs.deltag[self.mask!=0],
            yerr=self.inputs.deltag_error[self.mask!=0],
            fmt='o',
            capsize=2,
            label=self.get_label(),
            picker=picker if picker is not None else 5
        )       
        ax.errorbar(
            x=self.inputs.temperature[self.mask!=1],
            y=self.inputs.deltag[self.mask!=1],
            yerr=self.inputs.deltag_error[self.mask!=1],
            fmt='ro',
            capsize=2,
            label='_removed',
            picker=6
        )
              
        ax.legend(loc='best')
        ax.set_xlabel('Temperature (K)', fontsize=12)
        ax.set_ylabel(r'$\Delta$G$^\ddag$', fontsize=12)


    def plot_cpl_points(self, ax:'plt.Axes')->None:
        start = self.parameters.CpL_Point1.value - 1
        end = self.parameters.CpL_Point2.value
        ax.plot(
            self.inputs.temperature[self.mask!=0][start:end], 
            self.inputs.deltag[self.mask!=0][start:end], 
            'r.', 
            label='_o1_points',
            zorder=50
        )

    def get_label(self)->str:
        if self.name != '':
            return self.name
        return ' ' 
    
    def init_datapoint_mask(self)->None:
        if self.mask is None:
            self.mask = np.array([1 for i in self.inputs.temperature])

    def change_mask(self, idx:int, val:int)->None:
        if self.mask is None:
            self.init_datapoint_mask()
        self.mask[idx] = val

    def to_json(self)->str:
        out_json_dict = {
            'name': self.name,
            'inputs': {
                'temperature': list(self.inputs.temperature),
                'rate': list(self.inputs.rate),
                'deltag': list(self.inputs.deltag),
                'lnrate': list(self.inputs.lnrate),
                'rate_type': self.inputs.rate_type.value,
                'raw': [list(i) for i in list(self.inputs.raw)]
            },
            'mmrt2_parameters': {
                'H': self.parameters.H.to_json(),
                'S': self.parameters.S.to_json(),
                'ddH': self.parameters.ddH.to_json(),
                'CpH': self.parameters.CpH.to_json(),
                'Tn': self.parameters.Tn.to_json(),
                'Tc': self.parameters.Tc.to_json(),
                'CpL': self.parameters.CpL.to_json(),
                'CpL_Point1': self.parameters.CpL_Point1.to_json(),
                'CpL_Point2': self.parameters.CpL_Point2.to_json()
            },
            'mask': self.mask.tolist() if self.mask is not None else None
        }
        return json.dumps(out_json_dict, indent=1)
    
    @classmethod
    def from_json(cls, json_str:str):
        data = json.loads(json_str)
        name = data['name']
        inputs = data['inputs']
        mask = data['mask']
        parameters = None

        if 'mmrt2_parameters' in data.keys():
            parameters = data['mmrt2_parameters']

        return cls(
            name=name, 
            inputs=Inputs.from_json(inputs),
            parameters=MMRT2_Parameters.from_json(parameters),
            mask=np.array(mask) if mask is not None else None
        )


class Database(dict):
    def __init__(self, *args, **kwargs)->None:
        super(Database, self).__init__(*args, **kwargs)

    def add_datasets(self, datasets:list[Dataset])->None:
        for dataset in datasets:
            self.update({id(dataset): dataset})

    def remove_dataset(self, key:int)->None:
        del self[key]

    def get_dataset(self, key:int)->Dataset:
        return self.get(key)
    
    def get_keys(self)->list[int]:
        return self.keys()
    
    def get_datasets(self)->ValuesView[Dataset]:
        return self.values()
    
    def get_items(self)->ItemsView[int, Dataset]:
        return self.items()
    
    def get_length(self)->int:
        return len(self.values())

