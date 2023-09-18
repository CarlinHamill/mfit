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

from dataclasses import dataclass
import scipy
import numpy as np
import matplotlib.pyplot as plt

from typing import TYPE_CHECKING, Callable
if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from mfit.mfit2.data_tools import Dataset

@dataclass
class MMRTArgs:
    cl:float
    ch:float
    ddh:float
    tc:float
    tn:float
    s:float
    h:float

def heat_capacity(t:float|np.ndarray, args:MMRTArgs)->float|np.ndarray:
    cp = (args.cl + args.ch * np.exp((-args.ddh*(1-(t/args.tc)))/(8.314*t)))/(1 + np.exp((-args.ddh*(1-(t/args.tc)))/(8.314*t)))
    return cp

def integrate_equation(temperature_array:np.ndarray, integral_function:Callable, args:MMRTArgs)->np.ndarray:
    return np.array([scipy.integrate.quad(integral_function, args.tn, t, args=args)[0] for t in temperature_array])

def entropic_heat_capacity(t:float|np.ndarray, args:MMRTArgs)->float|np.ndarray:
    return heat_capacity(t, args)/t

def entropy(t:float|np.ndarray, args:MMRTArgs)->float|np.ndarray:
    s = args.s + integrate_equation(t, entropic_heat_capacity, args)
    return s

def enthalpy(t:float|np.ndarray, args:MMRTArgs)->float|np.ndarray:
    h = args.h + integrate_equation(t, heat_capacity, args)
    return h

def free_energy(t:float|np.ndarray, args:MMRTArgs)->float|np.ndarray:
    f = enthalpy(t, args) - t * entropy(t, args)
    return f

def log_k(t:float|np.ndarray, g:float|np.ndarray)->float|np.ndarray:
    return 23.76014 + np.log(t) - (g/(8.314 * t))

def rate(t:float|np.ndarray, g:float|np.ndarray)->float|np.ndarray:
    return np.exp(log_k(t, g))

def get_args(dataset:'Dataset')->MMRTArgs:
    if dataset.fit.mmrt2.mmrt2_fit is None:
        return None

    args = MMRTArgs(
        cl=dataset.fit.mmrt2.values['CpL'],
        ch=dataset.fit.mmrt2.values['CpH'],
        ddh=dataset.fit.mmrt2.values['ddH'],
        tc=dataset.fit.mmrt2.values['Tc'],
        tn=dataset.fit.mmrt2.values['Tn'],
        s=dataset.fit.mmrt2.values['S'],
        h=dataset.fit.mmrt2.values['H']
    )
    return args

def plot_deltag(axes:'plt.Axes', dataset:'Dataset', x:np.ndarray, g:np.ndarray)->None:
    axes.plot(dataset.inputs.temperature, dataset.inputs.deltag/1000, 'o', label=dataset.name)
    #lr = 23.76014 + np.log(x) - (g/(8.314 * x))
    axes.plot(x, g/1000, color=axes.get_lines()[-1].get_color())
    axes.legend(loc='best')

def plot_lnrate(ax:'plt.Axes', dataset:'Dataset', x:np.ndarray, lnk:np.ndarray)->None:
    ax.plot(dataset.inputs.temperature, dataset.inputs.lnrate, 'o', label=dataset.name)
    ax.plot(x, lnk, color=ax.get_lines()[-1].get_color())
    ax.legend(loc='best')

def plot_inverse(ax:'plt.Axes', dataset:'Dataset', x:np.ndarray, lnk:np.ndarray)->None:
    ax.plot(1/dataset.inputs.temperature, dataset.inputs.lnrate, 'o', label=dataset.name)
    ax.plot(1/x, lnk, color=ax.get_lines()[-1].get_color())
    ax.legend(loc='best')

def plot_rate(ax:'plt.Axes', dataset:'Dataset', x:np.ndarray, k:np.ndarray)->None:
    ax.plot(dataset.inputs.temperature, dataset.inputs.rate, 'o', label=dataset.name)
    ax.plot(x, k, color=ax.get_lines()[-1].get_color())
    ax.legend(loc='best')

def plot_enthalpy(ax:'plt.Axes', dataset:'Dataset', x:np.ndarray, h:np.ndarray)->None:
    ax.plot(x, h/1000, label=dataset.name)
    ax.legend(loc='best')

def plot_entropy(ax:'plt.Axes', dataset:'Dataset', x:np.ndarray, s:np.ndarray)->None:
    ax.plot(x, x*s/1000, label=dataset.name)
    ax.legend(loc='best')

def plot_heat_capacity(ax:'plt.Axes', dataset:'Dataset', x:np.ndarray, c:np.ndarray)->None:
    ax.plot(x, c/1000, label=dataset.name)
    ax.legend(loc='best')


def get_curves(dataset:'Dataset')->tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(dataset.inputs.temperature.min(), dataset.inputs.temperature.max(), 250)
    
    args = get_args(dataset)
    if args is None:
        return x, *[np.zeros(250) for _ in range(6)]

    g = free_energy(x, args)
    h = enthalpy(x, args)
    s = entropy(x, args)
    c = heat_capacity(x, args)
    
    return x, g, log_k(x, g), rate(x, g), h, s, c