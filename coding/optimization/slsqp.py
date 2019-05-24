"""wrapper for scipy SLSQP routine"""
from . import *
from scipy.optimize import minimize
import numpy as np


def neg_information(stimulus_pdf, sr_grid):
    return -information(sr_grid, stimulus_pdf)

def constr_pdf(p):
    return -p.sum() + 1

def constr_pos(p):
    return p

def constr_spike_count(p, sr_grid, max_n):
    out_pdf = p.dot(sr_grid)
    mean_out_count = out_pdf.dot(np.arange(len(out_pdf)))
    return max_n - mean_out_count

def optimize(sr_grid, init=None, eps=1e-3, max_iter=10000, max_spikes=200, verbose=False):
    n = sr_grid.shape[0]

    if init is None:
        new_pdf = np.ones(n)
        new_pdf = new_pdf / new_pdf.sum()
    elif all(init > 0) and np.isclose(np.sum(init), 1.):
        if len(init) == n:
            new_pdf = init
        else:
            raise ValueError('init has incorrect size. {} expected but len(init) is {}'.format(n, len(init)))
    else:
        raise ValueError('init is not a probability distribution')

    res = minimize(neg_information, new_pdf, args=(sr_grid,), method='SLSQP', constraints=(
        {'type': 'eq', 'fun': constr_pdf}, {'type': 'ineq', 'fun': constr_pos},
        {'type': 'ineq', 'fun': lambda x: constr_spike_count(x, sr_grid, max_spikes)}),
                    options={'maxiter': max_iter}, tol=eps)

    out_pdf = res['x'].dot(sr_grid)

    return {'pdf': res['x'], 'fun': -res['fun'], 'sensitivity': sensitivity(sr_grid, out_pdf), 'out_pdf': out_pdf}