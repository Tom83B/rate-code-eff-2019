from optimization import sensitivity, information
from scipy.optimize import minimize
# from scipy.signal import argrelmax
from scipy.ndimage.filters import gaussian_filter1d
from scipy.stats import entropy
import warnings
import copy
import numpy as np

def optimize(sr_grid, init=None, eps=1e-4, max_iter=100000, verbose=False, s=0, expense=None):
    n = sr_grid.shape[0]  # TODO sjednotit uvodni cast kodu s cutting plane

    if expense is None:
        expense = sr_grid.dot(np.arange(sr_grid.shape[1]))

    if init is None:
        new_pdf = np.ones(n)
        new_pdf = new_pdf / new_pdf.sum()
    elif all(init >= 0) and np.isclose(np.sum(init), 1.):
        if len(init) == n:
            new_pdf = init
        else:
            raise ValueError('init has incorrect size. {} expected but len(init) is {}'.format(n, len(init)))
    else:
        raise ValueError('init is not a probability distribution')


    for i in range(max_iter):
        prior = new_pdf.dot(sr_grid)
        c = np.exp(sensitivity(sr_grid, prior) - s * expense)
        min_exp_C = new_pdf.dot(c)
        new_pdf = new_pdf * c / min_exp_C

        min_capacity = np.log(min_exp_C)
        max_capacity = np.log(np.max(c))
        # capacity = (min_capacity + max_capacity) / 2
        capacity = min_capacity
        accuracy = (max_capacity - min_capacity)

        if verbose is True:
            print(f'min: {min_capacity}, max: {max_capacity}, accuracy: {accuracy}')

        if accuracy < eps:
            break
    else:
        warnings.warn(f'Maximum number of iterations ({max_iter}) reached', RuntimeWarning)

    fin_exp = new_pdf.dot(expense)
    capacity = min_capacity + s * fin_exp
    out_pdf = sr_grid.T.dot(new_pdf)


    return {'pdf': new_pdf, 'fun': capacity, 'out_pdf': out_pdf, 'sensitivity': sensitivity(sr_grid, out_pdf),
            'accuracy': accuracy, 'expense': fin_exp}

def relative_entropy(prob1, prob2):
    # prob1_smooth = gaussian_filter1d(prob1, sigma=0.1)
    prob1_smooth = prob1 + prob1[prob1 > 0].min() / 10
    prob1_smooth = prob1_smooth / prob1_smooth.sum()

    # prob2_smooth = gaussian_filter1d(prob2, sigma=0.1)
    prob2_smooth = prob2 + prob2[prob2 > 0].min() / 10
    prob2_smooth = prob2_smooth / prob2_smooth.sum()

    return (prob1_smooth * np.log(prob1_smooth / prob2_smooth)).sum()

def letter_derivative(letter, sr_func, min_intensity, max_intensity, out_prob):
    eps = (max_intensity - min_intensity) / 1000

    if letter - eps < min_intensity:
        try_letter = min_intensity
        left_shift = letter - min_intensity
    else:
        try_letter = letter - eps
        left_shift = eps

    prob1 = sr_func(try_letter)
    left_sensitivity = (prob1 * np.log2(prob1 / out_prob)).sum()
    # left_divergence = relative_entropy(prob1, out_prob)


    if letter + eps > max_intensity:
        try_letter = max_intensity
        right_shift = max_intensity - letter
    else:
        try_letter = letter + eps
        right_shift = eps

    prob1 = sr_func(try_letter)
    right_sensitivity = (prob1 * np.log2(prob1 / out_prob)).sum()
    # right_divergence = relative_entropy(prob1, out_prob)
    # print('r', right_divergence)
    # print(prob1)
    # print(out_prob)

    return (right_sensitivity - left_sensitivity) / (right_shift + left_shift)

def neg_sensitivity_sum(alphabet, sr_func, posterior):
    sr_grid = sr_func(alphabet)
    for i in range(sr_grid.shape[0]):
        sr_grid[i] += 1e-5
        sr_grid[i] = sr_grid[i] / sr_grid[i].sum()
    posterior += 1e-5
    posterior = posterior / posterior.sum()
    sensitivity = (sr_grid * np.log2(sr_grid / posterior)).sum(axis=1)
    return -sensitivity.sum()

def particle_optimization(sr_func, min_intensity, max_intensity, n_particles, n_steps=1000, **kwargs):
    eta = .1
    shift = (max_intensity - min_intensity) / 4
    alphabet = np.linspace(min_intensity, max_intensity, n_particles, endpoint=True)
    history = {
        'alphabet': [],
        'capacity': []
    }

    for i in range(n_steps):
        sr_grid = sr_func(alphabet)
        res = optimize(sr_grid, **kwargs)

        history['alphabet'].append(alphabet)
        history['capacity'].append(res['fun'])

        # for j in range(20):
        #     for k,letter in enumerate(alphabet):
        #         derivative = letter_derivative(letter, sr_func, min_intensity, max_intensity, res['out_pdf'])
        #         print(derivative, end=' ')
        #         alphabet[k] += derivative * eta
        #     print()
        #     alphabet = np.clip(alphabet, a_min=min_intensity, a_max=max_intensity)q1
        bounds = [(min_intensity, max_intensity) for i in range(len(alphabet))]
        alphabet = minimize(neg_sensitivity_sum, x0=alphabet, args=(sr_func, res['out_pdf']),
                            bounds=bounds, options={'maxiter': 20})['x']
        # print(alphabet)
        print(i)

    sr_grid = sr_func(alphabet)
    res = optimize(sr_grid, **kwargs)

    return res, alphabet, history

def alphabet_opt(sr_func, min_intensity, max_intensity, eps=1e-4, max_iter=100, ret_sequence=False):
    if ret_sequence:
        alphabet_seq = []

    # begin with two points - the boundary of the interval
    output_count = len(sr_func(min_intensity))

    alphabet = list(np.linspace(min_intensity, max_intensity, output_count))
    # sufficiently large grid should represent a continuous function
    tot_grid = np.linspace(min_intensity, max_intensity, num=1000, endpoint=True)#.tolist()

    accuracy = None

    for i in range(max_iter):
        sr_grid = sr_func(alphabet)
        res = optimize(sr_grid, eps=eps / 2)
        res['alphabet'] = copy.copy(alphabet)

        # prior = np.array([sr_func(c) * v for c,v in zip(alphabet, res['pdf'])]).sum(axis=0)
        prior = res['out_pdf']

        # prepare the function for computation of sensitivity
        # clip it in case it's infinite
        sensitivity_func = lambda x: np.clip(sensitivity(sr_func(x), prior), 0, 100)
        sensitivities = sensitivity_func(tot_grid)

        # local_maxima_ix = np.argwhere((
        #     (sensitivities[1:] > sensitivities[:-1]) &
        #     (sensitivities[1:] > Cn)
        # )).flatten().tolist()
        #
        # local_maxima = list(tot_grid[local_maxima_ix])
        #
        # if sensitivities[0] > sensitivities[1] and sensitivities[0] > Cn:
        #     local_maxima.append(0)
        # if sensitivities[-1] > sensitivities[-2] and sensitivities[-1] > Cn:
        #     local_maxima.append(len(sensitivities) - 1)
        #
        # local_max_sensitivities = sensitivity_func(local_maxima)
        #
        # local_maxima = [x for _, x in sorted(zip(local_max_sensitivities, local_maxima))][:1]
        # local_max_sensitivities = sorted(local_max_sensitivities)[:1]
        #
        # print(local_maxima)
        #
        # tot_alphabet = alphabet + local_maxima
        # all_sensitivities = np.concatenate((res['sensitivity'], local_max_sensitivities))
        #
        # sorted_inputs = [x for _, x in sorted(zip(all_sensitivities, tot_alphabet), reverse=True)]
        # alphabet = sorted_inputs[:sr_grid.shape[1]]

        # print(np.round(tot_alphabet, 5))
        # print(np.round(alphabet, 5))
        # print()

        # # add the point with highest sensitivity to alphabet
        best_intensity_index = sensitivities.argmax()
        best_intensity = tot_grid[best_intensity_index]
        largest_sensitivity = sensitivities[best_intensity_index]

        max_capacity = largest_sensitivity
        min_capacity = res['fun']
        capacity = min_capacity
        accuracy = (max_capacity - min_capacity) / capacity

        if ret_sequence:
            if accuracy is not None:
                alphabet_seq.append((res, sensitivities, accuracy))
            else:
                alphabet_seq.append((res, sensitivities, None))

        if accuracy < eps:
            break

        alphabet.append(best_intensity)

        # if the lenght of alphabet is greater or equal to the number of possible outputs
        # remove the intensity corresponding to lowest probability from alphabet
        if len(alphabet) >= sr_grid.shape[1]:
            low_index = np.argmin(res['sensitivity'])
            del alphabet[low_index]

        alphabet.sort()
    else:
        sr_grid = sr_func(alphabet)
        res = optimize(sr_grid)
        prior = res['out_pdf']

        sensitivity_func = lambda x: np.clip(sensitivity(sr_func(x), prior), 0, 100)
        sensitivities = sensitivity_func(tot_grid)

        if ret_sequence:
            alphabet_seq.append((copy.copy(alphabet), sensitivities, accuracy))

        warnings.warn(f'Maximum number of iterations ({max_iter}) reached', RuntimeWarning)

    res['alphabet'] = alphabet

    if ret_sequence:
        return res, alphabet_seq, tot_grid
    else:
        return res

def alphabet_opt(sr_func, min_intensity, max_intensity, s=0, expense=None, eps=1e-4, max_iter=15, ret_sequence=False,
                 init_alphabet=None):
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
        #
        # if sensitivities[0] > sensitivities[1] and sensitivities[0] > Cn:
        #     local_maxima.append(0)
        # if sensitivities[-1] > sensitivities[-2] and sensitivities[-1] > Cn:
        #     local_maxima.append(len(sensitivities) - 1)
        #
        # local_max_sensitivities = sensitivity_func(local_maxima)
        #
        # local_maxima = [x for _, x in sorted(zip(local_max_sensitivities, local_maxima))]
        # local_max_sensitivities = sorted(local_max_sensitivities)

        alphabet = [a for a, p in zip(alphabet, res['pdf']) if p > eps]
        alphabet = list(set(alphabet + local_maxima))
        # print(sensitivity_func(alphabet) - s * expense(alphabet))
        # all_sensitivities = np.concatenate((res['sensitivity'], local_max_sensitivities))
        #
        # sorted_inputs = [x for _, x in sorted(zip(all_sensitivities, tot_alphabet), reverse=True)]
        # alphabet = sorted_inputs[:sr_grid.shape[1]]

        # print(np.round(tot_alphabet, 5))
        # print(np.round(alphabet, 5))
        # print()

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


        # print(min_capacity)


        # if best_intensity not in alphabet:
        #     alphabet.append(best_intensity)
        # else:
        #     sr_grid = sr_func(alphabet)
        #     res = optimize(sr_grid, s=s)
        #     prior = res['out_pdf']
        #     res['alphabet'] = copy.copy(alphabet)
        #
        #     sensitivity_func = lambda x: np.clip(sensitivity(sr_func(x), prior), 0, 100)
        #     sensitivities = sensitivity_func(tot_grid)
        #
        #     if ret_sequence:
        #         alphabet_seq.append((res, sensitivities, accuracy))
        #
        #     break

        # if the lenght of alphabet is greater or equal to the number of possible outputs
        # remove the intensity corresponding to lowest probability from alphabet


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


# def information_derivatives(alphabet, weights, sr_func, min_intensity, max_intensity, eps):
#     derivatives = np.zeros(len(alphabet))
#
#     for i in range(len(alphabet)):
#         saved_val = alphabet[i]
#
#         if saved_val - eps < min_intensity:
#             alphabet[i] = min_intensity
#             left_shift = saved_val - min_intensity
#         else:
#             alphabet[i] = saved_val - eps
#             left_shift = eps
#
#         sr_grid = sr_func(alphabet)
#         left_information = information(sr_grid, weights)
#
#
#         if saved_val + eps > max_intensity:
#             alphabet[i] = max_intensity
#             right_shift = max_intensity - saved_val
#         else:
#             alphabet[i] = saved_val + eps
#             right_shift = eps
#
#         sr_grid = sr_func(alphabet)
#         right_information = information(sr_grid, weights)
#
#         derivatives[i] = (right_information - left_information) / (right_shift + left_shift)
#
#     return derivatives
#
# def update_alphabet(alphabet, weights, sr_func, min_intensity, max_intensity):
#     eps = (max_intensity - min_intensity) / 1000
#     derivatives = information_derivatives(alphabet, weights, sr_func, min_intensity, max_intensity)
#     new_alphabet = (alphabet + derivatives).clip(min=min_intensity, max=max_intensity)
#     return new_alphabet

if __name__ == '__main__':
    import numpy as np

    x = np.linspace(-0.5, 0.5, 10)
    y = np.linspace(-2, 2, 10000)

    arg_grid = np.subtract.outer(y, x).T

    rates = ((arg_grid >= -1) & (arg_grid <= 1))
    rates = rates / rates.sum(axis=1, keepdims=True)

    res = optimize(rates, max_iter=int(1e5), eps=1e-4)
    print('done')