import numpy as np
import copy
import warnings
from scipy.signal import argrelmax

from . import sensitivity
from .jimbo_kunisawa import optimize as jimbo_opt


def alphabet_opt_rel(sr_func, min_intensity, max_intensity,
                 expense=None, eps=1e-4, max_iter=15, ret_sequence=False, init_alphabet=None):
    if ret_sequence:
        alphabet_seq = []

    if expense is None:
        expense = lambda x: sr_func(x).dot(np.arange(sr_grid.shape[1]))

    # begin with two points - the boundary of the interval
    output_count = len(sr_func(min_intensity))

    if init_alphabet is None:
        alphabet = list([min_intensity, max_intensity])
    else:
        alphabet = copy.copy(init_alphabet)
    # sufficiently large grid should represent a continuous function
    tot_grid = np.linspace(min_intensity, max_intensity, num=1000, endpoint=True)#.tolist()

    accuracy = None
    res = None
    init = None

    for i in range(max_iter):
        sr_grid = sr_func(alphabet)

        if res is not None:
            print('alphabet', alphabet)
            print('past alphabet', res['alphabet'])
            init = list(res['pdf'][res['pdf'] > eps])
            for i, a in enumerate(alphabet):
                if a not in res['alphabet']:
                    print('a', a)
                    init = init[:i] + [0] + init[i:]
            init = np.array(init)
            init += len(init) / 100
            init = init / init.sum()
            print(init, init.sum())

        res = jimbo_opt(sr_grid, eps=eps / 10000, init=init)
        res['alphabet'] = copy.copy(alphabet)

        init = list(res['pdf'])


        # prior = np.array([sr_func(c) * v for c,v in zip(alphabet, res['pdf'])]).sum(axis=0)
        prior = res['out_pdf']

        # prepare the function for computation of sensitivity
        # clip it in case it's infinite
        sensitivity_func = lambda x: np.clip(sensitivity(sr_func(x), prior), 0, 100)
        sensitivities = sensitivity_func(tot_grid)
        rel_sensitivities = sensitivities / expense(tot_grid)

        local_maxima_ix = argrelmax(rel_sensitivities)
        if rel_sensitivities[-1] > rel_sensitivities[-2]:
            local_maxima_ix = np.append(local_maxima_ix, len(rel_sensitivities) - 1)
        if rel_sensitivities[0] > rel_sensitivities[1]:
            local_maxima_ix = np.append(local_maxima_ix, 0)

        local_maxima = list(tot_grid[local_maxima_ix])

        alphabet = [a for a, p in zip(alphabet, res['pdf']) if p > eps]
        alphabet = list(set(alphabet + local_maxima))

        # # add the point with highest sensitivity to alphabet
        best_intensity_index = rel_sensitivities.argmax()
        best_intensity = tot_grid[best_intensity_index]
        largest_sensitivity = rel_sensitivities[best_intensity_index]

        max_capacity = largest_sensitivity
        min_capacity = res['fun']
        capacity = min_capacity
        accuracy = (max_capacity - min_capacity)

        if ret_sequence:
            if accuracy is not None:
                alphabet_seq.append((res, rel_sensitivities, accuracy))
            else:
                alphabet_seq.append((res, rel_sensitivities, None))

        if accuracy < eps:
            break

        alphabet.sort()
    else:
        warnings.warn(f'Maximum number of iterations ({max_iter}) reached', RuntimeWarning)

    sr_grid = sr_func(alphabet)
    res = jimbo_opt(sr_grid, eps=eps / 1000)
    prior = res['out_pdf']
    res['alphabet'] = copy.copy(alphabet)

    if ret_sequence:
        sensitivity_func = lambda x: np.clip(sensitivity(sr_func(x), prior), 0, 100)
        sensitivities = sensitivity_func(tot_grid)
        alphabet_seq.append((res, sensitivities, accuracy))

    if ret_sequence:
        return res, alphabet_seq, tot_grid
    else:
        return res

def alphabet_opt(sr_func, min_intensity, max_intensity, optimize, sensitivity,
                 s=0, expense=None, eps=1e-4, max_iter=15, ret_sequence=False, init_alphabet=None):
    if ret_sequence:
        alphabet_seq = []

    if expense is None:
        expense = lambda x: sr_func(x).dot(np.arange(sr_grid.shape[1]))

    # begin with two points - the boundary of the interval
    output_count = len(sr_func(min_intensity))

    if init_alphabet is None:
        alphabet = list([min_intensity, max_intensity])
    else:
        alphabet = copy.copy(init_alphabet)
    # sufficiently large grid should represent a continuous function
    tot_grid = np.linspace(min_intensity, max_intensity, num=1000, endpoint=True)#.tolist()

    accuracy = None
    res = None
    init = None

    for i in range(max_iter):
        sr_grid = sr_func(alphabet)

        if res is not None:
            print('alphabet', alphabet)
            print('past alphabet', res['alphabet'])
            init = list(res['pdf'][res['pdf'] > eps])
            for i, a in enumerate(alphabet):
                if a not in res['alphabet']:
                    print('a', a)
                    init = init[:i] + [0] + init[i:]
            init = np.array(init)
            init += len(init) / 100
            init = init / init.sum()
            print(init, init.sum())

        res = optimize(sr_grid, eps=eps / 10000, s=s, init=init)
        res['alphabet'] = copy.copy(alphabet)

        init = list(res['pdf'])


        # prior = np.array([sr_func(c) * v for c,v in zip(alphabet, res['pdf'])]).sum(axis=0)
        prior = res['out_pdf']

        # prepare the function for computation of sensitivity
        # clip it in case it's infinite
        sensitivity_func = lambda x: np.clip(sensitivity(sr_func(x), prior), 0, 100)
        sensitivities = sensitivity_func(tot_grid) - s * expense(tot_grid)

        local_maxima_ix = argrelmax(sensitivities)
        if sensitivities[-1] > sensitivities[-2]:
            local_maxima_ix = np.append(local_maxima_ix, len(sensitivities) - 1)
        if sensitivities[0] > sensitivities[1]:
            local_maxima_ix = np.append(local_maxima_ix, 0)

        local_maxima = list(tot_grid[local_maxima_ix])

        alphabet = [a for a, p in zip(alphabet, res['pdf']) if p > eps]
        alphabet = list(set(alphabet + local_maxima))

        # # add the point with highest sensitivity to alphabet
        best_intensity_index = sensitivities.argmax()
        best_intensity = tot_grid[best_intensity_index]
        largest_sensitivity = sensitivities[best_intensity_index]

        max_capacity = largest_sensitivity
        min_capacity = res['fun'] - s * res['expense']
        capacity = min_capacity
        accuracy = (max_capacity - min_capacity)

        if ret_sequence:
            if accuracy is not None:
                alphabet_seq.append((res, sensitivities, accuracy))
            else:
                alphabet_seq.append((res, sensitivities, None))

        if accuracy < eps:
            break

        alphabet.sort()
    else:
        warnings.warn(f'Maximum number of iterations ({max_iter}) reached', RuntimeWarning)

    sr_grid = sr_func(alphabet)
    res = optimize(sr_grid, eps=eps / 1000, s=s)
    prior = res['out_pdf']
    res['alphabet'] = copy.copy(alphabet)

    if ret_sequence:
        sensitivity_func = lambda x: np.clip(sensitivity(sr_func(x), prior), 0, 100)
        sensitivities = sensitivity_func(tot_grid) - s * expense(tot_grid)
        alphabet_seq.append((res, sensitivities, accuracy))

    if ret_sequence:
        return res, alphabet_seq, tot_grid
    else:
        return res
