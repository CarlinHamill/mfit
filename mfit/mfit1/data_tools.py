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
import lmfit as lm
import math
import scipy
import numba
from enum import Enum
from dataclasses import dataclass, field
import numba
import json

from typing import TYPE_CHECKING, ValuesView, ItemsView, Self
if TYPE_CHECKING:
    import matplotlib.pyplot as plt


class RateType(Enum):    
    Rate = 0
    lnRate = 1
    DeltaG = 2


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
    value: float
    variable: bool

    def to_json(self)->str:
        return (self.value, self.variable)
    
@dataclass
class MMRT1_Parameters:
    H: Parameter = field(default_factory=lambda:Parameter(10000, True))
    C: Parameter = field(default_factory=lambda:Parameter(-1000, True))
    S: Parameter = field(default_factory=lambda:Parameter(-150, True))
    tn: Parameter = field(default_factory=lambda:Parameter(278.15, False))

    @classmethod
    def from_json(cls, mmrt1_parameters)->Self:
        if mmrt1_parameters is None:
            return cls()
        return cls(
            Parameter(mmrt1_parameters['H'][0], mmrt1_parameters['H'][1]),
            Parameter(mmrt1_parameters['C'][0], mmrt1_parameters['C'][1]),
            Parameter(mmrt1_parameters['S'][0], mmrt1_parameters['S'][1]),
            Parameter(mmrt1_parameters['tn'][0], mmrt1_parameters['tn'][1])
        )


@dataclass
class MMRT15_Parameters:
    H: Parameter = field(default_factory=lambda:Parameter(10000, True))
    M: Parameter = field(default_factory=lambda:Parameter(0, True))
    C: Parameter = field(default_factory=lambda:Parameter(-1000, True))
    S: Parameter = field(default_factory=lambda:Parameter(-150, True))
    tn: Parameter = field(default_factory=lambda:Parameter(278.15, False))
    
    @classmethod
    def from_json(cls, mmrt15_parameters)->Self:
        if mmrt15_parameters is None:
            return cls()
        return cls(
            Parameter(mmrt15_parameters['H'][0], mmrt15_parameters['H'][1]),
            Parameter(mmrt15_parameters['M'][0], mmrt15_parameters['M'][1]),
            Parameter(mmrt15_parameters['C'][0], mmrt15_parameters['C'][1]),
            Parameter(mmrt15_parameters['S'][0], mmrt15_parameters['S'][1]),
            Parameter(mmrt15_parameters['tn'][0], mmrt15_parameters['tn'][1])
        )

@dataclass
class MMRT_Parameters:
    mmrt1: MMRT1_Parameters = field(default_factory=MMRT1_Parameters)
    mmrt15: MMRT15_Parameters = field(default_factory=MMRT15_Parameters)

    @classmethod
    def from_json(cls, mmrt1_parameters, mmrt15_parameters)->Self:
        return cls(
            MMRT1_Parameters.from_json(mmrt1_parameters),
            MMRT15_Parameters.from_json(mmrt15_parameters)
        )


@numba.njit
def mmrt1_dg_eq(H:float, S:float, C:float, t:float, tn:float)->float:
    dg = H-t*S+C*(t-tn-t*np.log(t/tn))
    return dg


@numba.njit
def mmrt15_dg_eq(H:float, S:float, C:float, M:float, t:float, tn:float)->float:
    dg = H-t*S+C*(t-tn-t*np.log(t/tn))-(M/2)*((t-tn)**2)
    return dg


class FitMMRT1:
    def __init__(self, name:str, mmrt1_parameters:MMRT1_Parameters, inputs:Inputs, mask:np.ndarray) -> None:
        self.name = name
        self.mmrt1_parameters = mmrt1_parameters
        self.inputs = inputs
        self.mask = mask
        self.fit_mmrt1()

    @property
    def r_squared(self)->tuple[str]|None:
        if self.model_fit is None:
            return (0,0)
        rsq = 1 - self.model_fit.residual.var()/ self.inputs.deltag_all.var()
        n = self.model_fit.ndata
        p = self.model_fit.nvarys
        arsq = rsq - (1-rsq) * p / (n-p-1)
        return (rsq, arsq)

    @property
    def aicc(self)->float:
        if self.model_fit is None:
            return 0
        aic = self.model_fit.aic
        aicc = aic + (2*(self.model_fit.nvarys**2)+2*self.model_fit.nvarys)/(self.model_fit.ndata-self.model_fit.nvarys-1)
        return aicc

    def temperature_optimum(self)->float:
        c = self.values['C']
        h = self.values['H']
        tn = self.values['tn']

        mmrt1_topt = (c*tn - h)/(c + 8.314)
        mmrt1_tinf = (h-c*tn)/(-c+math.sqrt(-c*8.314))

        return (mmrt1_topt, mmrt1_tinf)

    @property
    def fit_statistics(self)->dict[str, float]:
        statistics = {
            'ndata': self.model_fit.ndata if self.model_fit is not None else 0,
            'nvarys': self.model_fit.nvarys if self.model_fit is not None else 0,
            'rsq': self.r_squared[0],
            'arsq': self.r_squared[1],
            'aic': self.model_fit.aic if self.model_fit is not None else 0,
            'aicc': self.aicc,
            'chi_sq': self.model_fit.chisqr if self.model_fit is not None else 0,
            'chi_sq_reduced': self.model_fit.redchi if self.model_fit is not None else 0,
            'bic': self.model_fit.bic if self.model_fit is not None else 0,
            'topt': self.temperature_optimum()[0] if self.model_fit is not None else 0,
            'tinf': self.temperature_optimum()[1] if self.model_fit is not None else 0
        }
        return statistics

    @staticmethod
    def calculate_dg_from_parameters(temperature_array:np.ndarray, params:dict|lm.Parameters)->np.ndarray:
        if isinstance(params, lm.Parameters):
            params = params.valuesdict()  

        H = params['H']
        S = params['S']
        C = params['C']
        tn = params['tn']

        return mmrt1_dg_eq(H, S, C, temperature_array, tn)

    def residual_to_minimize(self, parameters:lm.Parameters, temperature_array:np.ndarray, deltag_array:np.ndarray)->np.ndarray:
        return self.calculate_dg_from_parameters(temperature_array, parameters) - deltag_array

    def initialise_parameters(self)->lm.Parameters:
        pars = lm.Parameters()
        pars.add('H', value=self.mmrt1_parameters.H.value, vary=self.mmrt1_parameters.H.variable)
        pars.add('S', value=self.mmrt1_parameters.S.value, vary=self.mmrt1_parameters.S.variable)
        pars.add('C', value=self.mmrt1_parameters.C.value, vary=self.mmrt1_parameters.C.variable)
        pars.add('tn', value=self.mmrt1_parameters.tn.value, vary=self.mmrt1_parameters.tn.variable)
        return pars

    def fit_mmrt1(self)->None:
        parameters = self.initialise_parameters()
        self.minimizer = lm.Minimizer(
            self.residual_to_minimize,
            parameters,
            fcn_args=(self.inputs.temperature[self.mask!=0], self.inputs.deltag[self.mask!=0]),
            calc_covar=True
        )

        self.model_fit = self.minimizer.minimize(method='leastsq')

        self.values = self.model_fit.params.valuesdict()
        self.error = {
            'H': self.model_fit.params['H'].stderr, 
            'S': self.model_fit.params['S'].stderr, 
            'C': self.model_fit.params['C'].stderr,
            'tn': self.model_fit.params['tn'].stderr,
            'aic': self.model_fit.aic
            }
        
        self.calculate_confidence_interval()

    def calculate_confidence_interval(self)->None:
        """
        method from https://kitchingroup.cheme.cmu.edu/blog/2013/02/12/\
            Nonlinear-curve-fitting-with-parameter-confidence-intervals/#\
            :~:text=A%20confidence%20interval%20tells%20us,which%20best%20fit%20the%20data.
        """
        try:
            alpha = 0.05 # 1-0.05 => 95% CI
            n = len(self.inputs.temperature[self.mask!=0])
            nfix = len([par for par in self.model_fit.params if self.model_fit.params[par].vary == False])
            p = len(self.model_fit.params) - nfix
            dof = max(0, n - p)
            tval = scipy.stats.distributions.t.ppf(1.0-alpha/2., dof)
            keys = [par_name for par_name in self.model_fit.params.valuesdict().keys() if self.model_fit.params[par_name].vary == True]
            vals = [val for (par_name, val) in self.model_fit.params.valuesdict().items() if self.model_fit.params[par_name].vary == True]
            self.ci = {n: (p - (var**0.5)*tval, p + (var**0.5)*tval) for (n, p, var) in zip(keys, vals, np.diag(self.model_fit.covar))}

            ls =  ['H', 'S', 'C', 'tn']
            self.ci.update({key:'None' for key in ls if key not in self.ci.keys()})

        except ValueError:
            self.ci = {}
            ls =  ['H', 'S', 'C', 'tn']
            self.ci.update({key:'None' for key in ls if key not in self.ci.keys()})
        
    def plot(self, ax:'plt.Axes')->None:
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
            picker=5
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

    def calculate_fit_curve(self, plot_density=1000):
        x = np.linspace(min(self.inputs.temperature), max(self.inputs.temperature), plot_density)
        C = self.values['C']
        S = self.values['S']
        H = self.values['H']
        tn = self.mmrt1_parameters.tn.value
        deltag = (H-x*S+C*(x-tn-x*(np.log(x/tn))))
        return x, deltag 
    
    def get_label(self)->str:
        if self.name != '':
            return self.name
        return ' '


class FitMMRT15:
    def __init__(self, name:str, mmrt15_parameters:MMRT15_Parameters, inputs:Inputs, mask:np.ndarray) -> None:
        self.name = name
        self.mmrt15_parameters = mmrt15_parameters
        self.inputs = inputs
        self.mask = mask
        self.fit_mmrt15()
        
    @property
    def r_squared(self)->tuple[str]|None:
        if self.model_fit is None:
            return (0,0)
        rsq = 1 - self.model_fit.residual.var()/ self.inputs.deltag_all.var()
        n = self.model_fit.ndata
        p = self.model_fit.nvarys
        arsq = rsq - (1-rsq) * p / (n-p-1)
        return (rsq, arsq)

    @property
    def aicc(self)->float:
        if self.model_fit is None:
            return 0
        aic = self.model_fit.aic
        aicc = aic + (2*(self.model_fit.nvarys**2)+2*self.model_fit.nvarys)/(self.model_fit.ndata-self.model_fit.nvarys-1)
        return aicc

    def temperature_optimum(self)->tuple[float]:
        r = 8.314
        c = self.values['C']
        h = self.values['H']
        m = self.values['M']
        tn = self.values['tn']

        mmrt15_topt_1 = -(c + r)/m + math.sqrt(c**2 + 2*c*m*tn + 2*c*r - 2*h*m + m**2*tn**2 + r**2)/m
        mmrt15_topt_2 = -(c + r)/m - math.sqrt(c**2 + 2*c*m*tn + 2*c*r - 2*h*m + m**2*tn**2 + r**2)/m
        mmrt15_tinf = (2*c*tn-2*h+m*(tn**2))/(c+r)
        
        return (mmrt15_topt_1, mmrt15_topt_2, mmrt15_tinf)

    @property
    def fit_statistics(self)->dict[str, float]:
        statistics = {
            'ndata': self.model_fit.ndata if self.model_fit is not None else 0,
            'nvarys': self.model_fit.nvarys if self.model_fit is not None else 0,
            'rsq': self.r_squared[0],
            'arsq': self.r_squared[1],
            'aic': self.model_fit.aic if self.model_fit is not None else 0,
            'aicc': self.aicc,
            'chi_sq': self.model_fit.chisqr if self.model_fit is not None else 0,
            'chi_sq_reduced': self.model_fit.redchi if self.model_fit is not None else 0,
            'bic': self.model_fit.bic if self.model_fit is not None else 0,
            'topt': f'{self.temperature_optimum()[0]:.2f}, {self.temperature_optimum()[1]:.2f}' if self.model_fit is not None else 0,
            'tinf': self.temperature_optimum()[2] if self.model_fit is not None else 0
        }
        return statistics
    
    @staticmethod
    def calculate_dg_from_parameters(temperature_array:np.ndarray, params:dict|lm.Parameters)->np.ndarray:
        if isinstance(params, lm.Parameters):
            params = params.valuesdict()  

        H = params['H']
        S = params['S']
        C = params['C']
        M = params['M']
        tn = params['tn']

        return mmrt15_dg_eq(H, S, C, M, temperature_array, tn)

    def residual_to_minimize(self, parameters:lm.Parameters, temperature_array:np.ndarray, deltag_array:np.ndarray)->np.ndarray:
        return self.calculate_dg_from_parameters(temperature_array, parameters) - deltag_array

    def initialise_parameters(self)->lm.Parameters:
        pars = lm.Parameters()
        pars.add('H', value=self.mmrt15_parameters.H.value, vary=self.mmrt15_parameters.H.variable)
        pars.add('S', value=self.mmrt15_parameters.S.value, vary=self.mmrt15_parameters.S.variable)
        pars.add('C', value=self.mmrt15_parameters.C.value, vary=self.mmrt15_parameters.C.variable)
        pars.add('M', value=self.mmrt15_parameters.M.value, vary=self.mmrt15_parameters.M.variable)
        pars.add('tn', value=self.mmrt15_parameters.tn.value, vary=self.mmrt15_parameters.tn.variable)
        return pars

    def fit_mmrt15(self)->None:
        parameters = self.initialise_parameters()
        self.minimizer = lm.Minimizer(
            self.residual_to_minimize,
            parameters,
            fcn_args=(self.inputs.temperature[self.mask!=0], self.inputs.deltag[self.mask!=0]),
            calc_covar=True
        )

        self.model_fit = self.minimizer.minimize(method='leastsq')

        self.values = self.model_fit.params.valuesdict()
        self.error = {
            'H': self.model_fit.params['H'].stderr, 
            'S': self.model_fit.params['S'].stderr, 
            'C': self.model_fit.params['C'].stderr,
            'M': self.model_fit.params['M'].stderr,
            'tn': self.model_fit.params['tn'].stderr,
            'aic': self.model_fit.aic
            }
        
        self.calculate_confidence_interval()

    def calculate_confidence_interval(self)->None:
        """
        method from https://kitchingroup.cheme.cmu.edu/blog/2013/02/12/\
            Nonlinear-curve-fitting-with-parameter-confidence-intervals/#\
            :~:text=A%20confidence%20interval%20tells%20us,which%20best%20fit%20the%20data.
        """
        try:
            alpha = 0.05 # 1-0.05 => 95% CI
            n = len(self.inputs.temperature[self.mask!=0])
            nfix = len([par for par in self.model_fit.params if self.model_fit.params[par].vary == False])
            p = len(self.model_fit.params) - nfix
            dof = max(0, n - p)
            tval = scipy.stats.distributions.t.ppf(1.0-alpha/2., dof)
            keys = [par_name for par_name in self.model_fit.params.valuesdict().keys() if self.model_fit.params[par_name].vary == True]
            vals = [val for (par_name, val) in self.model_fit.params.valuesdict().items() if self.model_fit.params[par_name].vary == True]
            self.ci = {n: (p - (var**0.5)*tval, p + (var**0.5)*tval) for (n, p, var) in zip(keys, vals, np.diag(self.model_fit.covar))}

            ls =  ['H', 'S', 'C', 'M', 'tn']
            self.ci.update({key:'None' for key in ls if key not in self.ci.keys()})

        except ValueError:
            self.ci = {}
            ls =  ['H', 'S', 'C', 'M', 'tn']
            self.ci.update({key:'None' for key in ls if key not in self.ci.keys()})


    def plot(self, ax:'plt.Axes')->None:
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
            picker=5
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

    def calculate_fit_curve(self, plot_density=1000):
        x = np.linspace(min(self.inputs.temperature), max(self.inputs.temperature), plot_density)
        C = self.values['C']
        S = self.values['S']
        H = self.values['H']
        M = self.values['M']
        tn = self.mmrt15_parameters.tn.value
        deltag = (H-x*S+C*(x-tn-x*(np.log(x/tn))))-(M/2)*((x-tn)**2)
        return x, deltag
    
    def get_label(self)->str:
        if self.name != '':
            return self.name
        return ' '


@dataclass
class FitMMRT:
    mmrt1: FitMMRT1 = None
    mmrt15: FitMMRT15 = None

    def fit_mmrt(self, name:str, parameters:MMRT_Parameters, inputs:Inputs, mask:np.ndarray)->None:
        self.fit_mmrt1(name, parameters.mmrt1, inputs, mask)
        self.fit_mmrt15(name, parameters.mmrt15, inputs, mask)

    def fit_mmrt1(self, name:str, mmrt1_parameters:MMRT1_Parameters, inputs:Inputs, mask:np.ndarray)->None:
        try:
            self.mmrt1 = FitMMRT1(name, mmrt1_parameters, inputs, mask)
        except Exception:
            self.mmrt1 = None
            self.mmrt1.model_fit = None
            self.mmrt1_inputs = inputs

    def fit_mmrt15(self, name:str, mmrt15_parameters:MMRT15_Parameters, inputs:Inputs, mask:np.ndarray)->None:
        try:
            self.mmrt15 = FitMMRT15(name, mmrt15_parameters, inputs, mask)
        except Exception:
            self.mmrt15 = None
            self.mmrt15.model_fit = None
            self.mmrt1_inputs = inputs

    def get_fit_curve(self)->tuple[np.ndarray]:
        mmrt1 = self.mmrt1.calculate_fit_curve(plot_density=250) if self.mmrt1 is not None else (np.linspace(min(self.mmrt1_inputs.temperature), max(self.mmrt1_inputs.temperature), 250), np.zeros(250))
        mmrt15 = self.mmrt15.calculate_fit_curve(plot_density=250)[1] if self.mmrt15 is not None else (np.zeros(250))
        return mmrt1 + (mmrt15,)


@dataclass
class Dataset:
    name:str
    inputs: Inputs
    parameters: MMRT_Parameters = field(default_factory=MMRT_Parameters)
    fit: FitMMRT = field(default_factory=FitMMRT)
    mask:np.ndarray = None

    def fit_mmrt(self)->None:
        if self.mask is None:
            self.init_datapoint_mask() 
    
        self.fit.fit_mmrt(self.name, self.parameters, self.inputs, self.mask)

    def init_datapoint_mask(self)->None:
        self.mask = np.array([1 for i in self.inputs.temperature])
    
    def change_input_type(self, new_rate_type:int)->None:
        temperature, rate_data, raw_array = self.inputs.get_input_rate()

        if new_rate_type == 0:
            self.inputs = Inputs.from_rate(temperature, rate_data, raw_array)
        if new_rate_type == 1:
            self.inputs = Inputs.from_lnrate(temperature, rate_data, raw_array)
        if new_rate_type == 2:
            self.inputs = Inputs.from_deltag(temperature, rate_data, raw_array)
        
        self.fit_mmrt()

    def plot(self, ax:'plt.Axes')->None:      
        ax.errorbar(
            x=self.inputs.temperature[self.mask!=0],
            y=self.inputs.deltag[self.mask!=0],
            yerr=self.inputs.deltag_error[self.mask!=0],
            fmt='o',
            capsize=2,
            label=self.name,
            picker=5
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

    def change_name(self, new_name:str)->None:
        self.name = new_name
        self.fit.mmrt1.name = new_name
        self.fit.mmrt15.name = new_name

    def change_mask(self, idx:int, val:int)->None:
        if self.mask is None:
            self.init_datapoint_mask()
        
        self.mask[idx] = val
        self.fit_mmrt()

    @classmethod
    def from_json(cls, json_str:str):
        data = json.loads(json_str)
        name = data['name']
        inputs = data['inputs']
        mask = data['mask']
        mmrt1_parameters = None
        mmrt15_parameters = None

        if 'mmrt1_parameters' in data.keys():
            mmrt1_parameters = data['mmrt1_parameters']
        if 'mmrt15_parameters' in data.keys():
            mmrt15_parameters = data['mmrt15_parameters']

        return cls(
            name=name, 
            inputs=Inputs.from_json(inputs),
            parameters=MMRT_Parameters.from_json(mmrt1_parameters, mmrt15_parameters),
            mask=np.array(mask) if mask is not None else None
        )
    
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
            'mmrt1_parameters': {
                'H': self.parameters.mmrt1.H.to_json(),
                'C': self.parameters.mmrt1.C.to_json(),
                'S': self.parameters.mmrt1.S.to_json(),
                'tn': self.parameters.mmrt1.tn.to_json(),
            },
            'mmrt15_parameters': {
                'H': self.parameters.mmrt15.H.to_json(),
                'M': self.parameters.mmrt15.M.to_json(),
                'C': self.parameters.mmrt15.C.to_json(),
                'S': self.parameters.mmrt15.S.to_json(),
                'tn': self.parameters.mmrt15.tn.to_json(),
            },
            'mask': self.mask.tolist() if self.mask is not None else None                     
        }
        return json.dumps(out_json_dict, indent=1)

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