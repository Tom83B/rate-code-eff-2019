from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np
import matplotlib.pyplot as plt
from typing import List

from functools import wraps

def plotter(plotting_func):
    @wraps(plotting_func)
    def wrapper_function(*args, **kwargs):
        if 'ax' not in kwargs.keys():
            fig, ax = plt.subplots()
            kwargs['ax'] = ax
        return plotting_func(*args, **kwargs)
    return wrapper_function

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']


def stimulus_response_hm(intensities, rates, ax=None, cmap='Blues', **kwargs):
    retf = False
    if ax is None:
        f, ax = plt.subplots()

    n_responses = rates.shape[1]
    # aspect = 0.8 * (intensities[-1] - intensities[0]) / n_responses
    aspect = 'auto'
    img = ax.imshow(rates.T, cmap=cmap, origin='lower',
              extent=[intensities[0], intensities[-1], 0, n_responses],
              aspect=aspect, **kwargs)
    return img

@plotter
def pmf_plot(pmf, *, ax=None, **kwargs):
    outs = np.arange(len(pmf))
    ax.fill_between(outs, pmf, step="pre", alpha=0.4)

@plotter
def stimulus_response_line(intensities, rates, color='C0', label='', *, ax=None, **kwargs):
    outs = np.arange(rates.shape[1])
    mu = rates.dot(outs)
    std = np.sqrt(rates.dot(outs * outs) - mu ** 2)

    intensities = np.linspace(intensities[0], intensities[-1], len(intensities))

    ax.plot(intensities, mu, label=label, c=color, **kwargs)
    ax.plot(intensities, mu + std, linestyle='dashed', c=color, **kwargs)
    ax.plot(intensities, mu - std, linestyle='dashed', c=color, **kwargs)

# def stimulus_response_line(intensities, rates, ax=None):
#     def moment(pdf, n, central=False):
#         if central:
#             c = moment(pdf, 1, central=False)
#         else:
#             c = 0
#         tot = 0
#         for i, p in enumerate(pdf):
#             tot += p * ((i - c) ** n)
#         return tot
#
#     retf = False
#     if ax is None:
#         f, ax = plt.subplots()
#
#     means = np.array([moment(pdf, 1) for pdf in rates])
#     stds = np.array([np.sqrt(moment(pdf, 2, central=True)) for pdf in rates])
#
#     ax.errorbar(intensities, means, yerr=stds)
#     return ax

def voltage_plot(sim: 'simulations.simulation.NeuronSimulation', ax, t_range=None, **kwargs):
    """Plots the membrane voltage and voltage threshol

    :param sim: NeuronSimulation object
    :param t_range: tuple with time range to be plotted
    :param ax: where to plot it
    """
    if not t_range:
        t_range = (sim.time_array.min(), sim.time_array.max())

    mask = (sim.time_array >= t_range[0]) & (sim.time_array < t_range[1])
    ax.plot(sim.time_array[mask], sim.voltage_array[mask], **kwargs)
    ax.plot(sim.time_array[mask], sim.threshold_array[mask])


def spike_voltage_plot(sim: 'simulations.simulation.NeuronSimulation', ax, t_range=None):
    if not t_range:
        t_range = (sim.time_array.min(), sim.time_array.max())

    mask = (sim.time_array >= t_range[0]) & (sim.time_array < t_range[1])
    print(mask)

    time = sim.time_array[mask]
    voltage = sim.voltage_array[mask]
    v_func = InterpolatedUnivariateSpline(time, voltage, k=1)

    ax.plot(time, voltage)

    lo = voltage.min() - 5.
    hi = (voltage.max() - voltage.min()) * 3 + lo
    range = hi - lo
    ax.set_ylim(lo, hi)

    for spike_time in sim.spike_times[(sim.spike_times >= t_range[0]) & (sim.spike_times < t_range[1])]:
        lo_ratio = (v_func(spike_time) - lo) / range
        ax.axvline(spike_time, lo_ratio, 0.9)


def current_plot(sim: 'simulations.simulation.NeuronSimulation', ax, t_range=None):
    """Plots the membrane voltage and voltage threshol

    :param sim: NeuronSimulation object
    :param t_range: tuple with time range to be plotted
    :param ax: where to plot it
    """
    if not t_range:
        t_range = (sim.time_array.min(), sim.time_array.max())

    mask = (sim.time_array >= t_range[0]) & (sim.time_array < t_range[1])
    ax.plot(sim.time_array[mask], sim.current_array[mask])


def spike_plot(sim: 'simulations.simulation.NeuronSimulation', lo, hi, ax):
    for t in sim.spike_times:
        ax.axvline(t)


def voltage_threshold_intensity(sims: List['simulations.simulation.NeuronSimulation'], inputs, indexes, ax, t_range):
    for i, ix in enumerate(indexes):
        sim = sims[ix]
        mask = (sim.time_array >= t_range[0]) & (sim.time_array < t_range[1])
        ax.plot(sim.time_array[mask], sim.voltage_array[mask], label=str(inputs[ix]) + ' Hz', c=colors[i])
        ax.plot(sim.time_array[mask], sim.threshold_array[mask], c=colors[i], linestyle='-.')

    ax.set_ylabel('V (mV)')
    ax.set_xlabel('t (ms)')
    ax.set_xrange(t_range)
    ax.legend(loc='best')

def voltage_threshold_intensity_single(sim: 'simulations.simulation.NeuronSimulation', ax, t_range):
    mask = (sim.time_array >= t_range[0]) & (sim.time_array < t_range[1])
    ax.plot(sim.time_array[mask], sim.voltage_array[:-1][mask])
    ax.plot(sim.time_array[mask], sim.threshold_array[:-1][mask], linestyle='-.')

    ax.set_ylabel('V (mV)')
    ax.set_xlabel('t (ms)')
    ax.set_xlim(t_range)

def plot_discrete_distribution(intensities, pdf, ax, cutoff=1e-10, color='C0', label_axes=True):
    """Plots probability distribution with discrete support

    :param intensities: input / stimulus intensities - for x-axis specifications
    :param pdf: the probability distribution
    :param ax: where to plot it
    :param cutoff: everything in the pdf with value lower then cutoff will be considered 0
    """
    hi = pdf.max() * 1.2
    ax.set_ylim((0, hi))
    ax.set_xlim((intensities.min() * 0.8, intensities.max() + intensities.min()))

    if label_axes == True:
        ax.set_xlabel('f (Hz)')
        ax.set_ylabel('PDF')

    mask = np.zeros_like(intensities, dtype=bool)

    for i, (intensity, p) in enumerate(zip(intensities, pdf)):
        if p > cutoff:
            ax.axvline(intensity, 0, p / hi, c=color)
            mask[i] = True

    ax.scatter(intensities[mask], pdf[mask], c=color)

def sensitivity_plot(intensities, sensitivity, ax, cutoff=1e-10, pdf_marks=True, pdf=None):
    if pdf_marks == True:
        if pdf is not None:
            for intensity, p in zip(intensities, pdf):
                if p > cutoff:
                    ax.axvline(intensity, 0, 1, c='orange', linewidth=0.5)
        else:
            raise TypeError('pdf has to be given, if pdf_marks is set to True')

    ax.set_xlim((0, intensities.max() + intensities.min()))
    ax.plot(intensities, sensitivity)
