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

from typing import TYPE_CHECKING, Callable
if TYPE_CHECKING:
    from mfit.mfit1.data_tools import Dataset
    import matplotlib.pyplot as plt

@dataclass
class MMRTArgs:
    c:float
    s:float
    h:float
    tn:float
    m:float=0

def heat_capacity(t:float|np.ndarray, args:MMRTArgs)->float|np.ndarray:
    cp = args.c + args.m * t
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

def get_mmrt1_args(dataset:'Dataset')->MMRTArgs:
    if dataset.fit.mmrt1 is None:
        return None

    args = MMRTArgs(
        c=dataset.fit.mmrt1.values['C'],
        s=dataset.fit.mmrt1.values['S'],
        h=dataset.fit.mmrt1.values['H'],
        tn=dataset.fit.mmrt1.values['tn']
    )
    return args

def get_mmrt15_args(dataset:'Dataset')->MMRTArgs:
    if dataset.fit.mmrt15 is None:
        return None

    args = MMRTArgs(
        c=dataset.fit.mmrt15.values['C'],
        s=dataset.fit.mmrt15.values['S'],
        h=dataset.fit.mmrt15.values['H'],
        tn=dataset.fit.mmrt15.values['tn'],
        m=dataset.fit.mmrt15.values['M']
    )
    return args

def plot_deltag(axes:'plt.Axes', dataset:'Dataset', x:np.ndarray, g:np.ndarray)->None:
    axes.plot(dataset.inputs.temperature, dataset.inputs.deltag/1000, 'o', label=dataset.name)
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

def get_curves(dataset: 'Dataset')->tuple[np.ndarray]:
    x = np.linspace(dataset.inputs.temperature.min(), dataset.inputs.temperature.max(), 250)

    mmrt1_args = get_mmrt1_args(dataset)
    if mmrt1_args is None:
        mmrt1_curves = [np.zeros(250) for _ in range(6)]
    else:
        g1 = free_energy(x, mmrt1_args)
        h1 = enthalpy(x, mmrt1_args)
        s1 = entropy(x, mmrt1_args)
        c1 = heat_capacity(x, mmrt1_args)
        mmrt1_curves = (g1, log_k(x, g1), rate(x, g1), h1, s1, c1)
    mmrt15_args = get_mmrt15_args(dataset)
    if mmrt15_args is None:
        mmrt15_curves = [np.zeros(250) for _ in range(6)]
    else:
        g15 = free_energy(x, mmrt15_args)
        h15 = enthalpy(x, mmrt15_args)
        s15 = entropy(x, mmrt15_args)
        c15 = heat_capacity(x, mmrt15_args)
        mmrt15_curves = (g15, log_k(x, g15), rate(x, g15), h15, s15, c15)

    return x, *mmrt1_curves, *mmrt15_curves

    
