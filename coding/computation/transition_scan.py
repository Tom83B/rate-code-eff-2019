from pyneuron import ShotNoiseConductance, MATThresholds, Neuron, sr_experiment, steady_spike_train
from neurons import get_mat
import numpy as np
import pandas as pd
from itertools import product
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

reshaped_results = {}
time_windows = [250, 500, 750, 1000]

exc = ShotNoiseConductance(
    rate=2.67,
    g_peak=0.0015,
    reversal=0,
    decay=3)

inh = ShotNoiseConductance(
    rate=3.73,
    g_peak=0.0015,
    reversal=-75,
    decay=10)


def intensity_freq_func(intensity, B):
    exc = 2.67 * intensity
    inh = 3.73 * (1 + B * (intensity - 1))
    return exc, inh

def get_results(seed):
    # for B in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    for B in [0]:
        if seed == 0:
            print(f'B = {B:.1f}')
            mtqdm = tqdm
        else:
            mtqdm = lambda x: x
        for int1, int2 in mtqdm(list(product(np.logspace(0, 1.6, 100), repeat=2))):
            RS = get_mat('RS')
            IB = get_mat('IB')
            FS = get_mat('FS')
            CH = get_mat('CH')

            neuron = Neuron(
                resting_potential=-80,
                membrane_resistance=50,
                membrane_capacitance=0.1,
                mats=[RS, IB, FS, CH])

            neuron.append_conductance(exc)
            neuron.append_conductance(inh)

            func = partial(intensity_freq_func, B=B)

            intensities = [int1, int2] * 100
            res = sr_experiment(neuron, time_windows, 0.1, intensities, func, seed)

            for tw in time_windows:
                for neuron_name in ['RS','IB','FS','CH']:
                    reshaped_results[(B, tw, neuron_name, int1, int2)] = res[tw].loc[int1, neuron_name]
                    reshaped_results[(B, tw, neuron_name, int2, int1)] = res[tw].loc[int2, neuron_name]

    return pd.Series(reshaped_results).unstack()


def sersum(serlist):
    tmp = serlist[0]
    for ser in serlist[1:]:
        tmp = tmp + ser
    return tmp

n_jobs = cpu_count()
p = Pool(n_jobs)
result_list = p.map(get_results, range(n_jobs))
results_summed = sersum(result_list)
results_summed.to_pickle('transition_scan.pkl')