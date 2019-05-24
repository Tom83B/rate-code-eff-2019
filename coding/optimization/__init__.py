import numpy as np


def sensitivity(sr_grid, prior):
    # prior = stimulus_pdf.dot(sr_grid)
    with np.errstate(divide='ignore'):
        specificity = sr_grid / prior
        specificity[np.isnan(specificity)] = 1
        return np.nan_to_num((sr_grid * np.nan_to_num(np.log2(specificity))).sum(axis=1))

def sensitivity_func(sr_func, prior):
    def specificity(x):
        spec = sr_func(x) / prior
        spec[np.isnan(spec)] = 1
        return spec
    specificity = lambda x: sr_func(x) / prior

def information(sr_grid, stimulus_pdf):
    posterior = stimulus_pdf.dot(sr_grid)
    return sensitivity(sr_grid, posterior).dot(stimulus_pdf)