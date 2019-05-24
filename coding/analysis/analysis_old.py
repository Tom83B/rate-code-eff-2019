import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from analysis.plotting import voltage_plot, voltage_threshold_intensity
from simulations.simulation import NeuronSimulation, zeros_func
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline


def split_by_time_window(spike_times, time_window):
    spikes_split = []
    max_time = max(spike_times)

    for t0 in np.arange(0, max_time, time_window):
        spikes_in_window = spike_times[(spike_times > t0) & (spike_times <= t0 + time_window)]
        spikes_split.append(spikes_in_window)

def spike_count_by_time_window(spike_times, time_window, max_time):
    spikes_split = []

    for t0 in np.arange(0, max_time, time_window):
        spikes_in_window = spike_times[(spike_times > t0) & (spike_times <= t0 + time_window)]
        spikes_split.append(len(spikes_in_window))

    return spikes_split

def convert_to_grid(input_frequencies, spike_counts):
    max_count = max([max(x[1:]) for x in spike_counts])

    grid = np.zeros(shape=(len(input_frequencies), max_count+1))

    for i, sc in enumerate(spike_counts):
        for count in sc[1:]:
            grid[i, count] += 1.

    return grid

def interpolation(grid, input_frequencies):
    x = input_frequencies
    y = np.arange(0, grid.shape[1], 1)

    new_y = np.linspace(y.min(), y.max(), 1000)

    grid_pdf = np.array(grid, copy=True, dtype=float)

    for i in range(len(x)):
        f = InterpolatedUnivariateSpline(y, grid[i,:], k=1, ext=1)
        tot = f(new_y).sum()
        grid_pdf[i,:] /=  tot

    plt.imshow(grid_pdf)

    return RectBivariateSpline(x, y, grid_pdf, kx=1, ky=1)


if __name__ == '__main__':
    f = open('mat.p', 'r')
    simulations = pickle.load(f)
    f.close()

    input_frequencies = simulations['inputs']
    simulations = simulations['simulations']
    spike_times_data = [x.spikeTimes for x in simulations]

    time_window = 100
    max_time = 5000.
    spike_counts = [spike_count_by_time_window(x, time_window, max_time) for x in spike_times_data]

    means = [np.mean(x[1:]) * 10 for x in spike_counts]
    stds = [np.std(x[1:]) * 10 for x in spike_counts]


    plt.errorbar(x=input_frequencies, y=means, yerr=stds)
    plt.xlabel('input freq (Hz)')
    plt.ylabel('output freq (Hz)')

    # grid = convert_to_grid(input_frequencies, spike_counts)

    print('done')