from cvxopt import solvers, matrix
from optimization import sensitivity
import numpy as np
import warnings


def update_sensitivity_matrix(old_matrix, sr_grid, new_stimulus_pdf):
    new_sensitivity = sensitivity(sr_grid, new_stimulus_pdf)
    lim = 1e4
    new_sensitivity[new_sensitivity > lim] = lim
    new_sensitivity[new_sensitivity < -lim] = -lim
    vec = np.ones(len(new_sensitivity)+1)
    vec[1:] = -new_sensitivity
    return np.concatenate((old_matrix, [vec]), axis=0)

def optimize(sr_grid, expense, W, init=None, eps=1e-2, verbose=False, max_iter=10000):
    solvers.options['show_progress'] = False
    solvers.options['glpk'] = dict(msg_lev='GLP_MSG_OFF')

    n = sr_grid.shape[0]
    d = np.zeros(n+1)
    d[0] = 1.


    # A1 ... sum(p) = 1
    A1 = np.ones((1,n+1))
    A1[0,0] = 0.
    A1 = A1
    b = np.array([1.])

    stimulus_pdfs = []

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

    stimulus_pdfs.append(new_pdf)
    sensitivity_matrix = np.zeros((max_iter, n+1))
    sensitivity_matrix[:,0] = 1.

    # G0 ... p > 0
    G0 = -np.eye(sr_grid.shape[0], sr_grid.shape[0] + 1, 1, dtype=float)
    cost_matrix = np.concatenate(([[0]],[expense]), axis=1)
    
    G = np.concatenate((cost_matrix, G0, sensitivity_matrix))
#     G = np.concatenate((G0, sensitivity_matrix))

    h = np.zeros(G.shape[0], dtype=float)
    h[0] = W

    min_capacity = 0
    for i in range(max_iter):
        prior = sr_grid.T.dot(stimulus_pdfs[i])
        new_sensitivity = sensitivity(sr_grid, prior)
        lim = 1e2
        new_sensitivity[new_sensitivity > lim] = lim
        sensitivity_matrix[i,1:] = -new_sensitivity
#         G[G0.shape[0] + i, 1:] = -new_sensitivity
        G[G0.shape[0] + i + 1, 1:] = -new_sensitivity
        
#         row_count = G0.shape[0] + i + 1
        row_count = G0.shape[0] + i + 2

        if i >= 0:
            res = solvers.lp(matrix(-d), matrix(G[:row_count]), matrix(h[:row_count]), matrix(A1), matrix(b), solver=None)
        else:
            res = solvers.lp(-d, G[:row_count], h[:row_count], A1, b, solver=None)
        
#         print(res)
        
        stimulus_pdfs.append(np.array(res['x']).T[0,1:])
        
        
        prior = sr_grid.T.dot(stimulus_pdfs[-1])
        new_sensitivity = sensitivity(sr_grid, prior)
        lim = 1e2
        new_sensitivity[new_sensitivity > lim] = lim
        
        min_capacity = (new_sensitivity * stimulus_pdfs[-1]).sum()
        max_capacity = -res['primal objective']
#         max_capacity = stimulus
        capacity = min_capacity

        if min_capacity > 0:
            accuracy = (max_capacity - min_capacity) / capacity
        else:
            accuracy = np.inf

        if verbose is True:
            print('min: {}, max: {}, accuracy: {}'.format(min_capacity, max_capacity,
                                                          (max_capacity - min_capacity) / min_capacity))
        if accuracy < eps:
            break
    else:
        warnings.warn(f'Maximum number of iterations ({max_iter}) reached', RuntimeWarning)

    capacity = (min_capacity + max_capacity) / 2
    out_pdf = sr_grid.T.dot(stimulus_pdfs[-1])

    return {'pdf': stimulus_pdfs[-1], 'fun': capacity, 'sensitivity': sensitivity(sr_grid, out_pdf)}

if __name__ == '__main__':

    x = np.linspace(-0.5, 0.5, 10)
    y = np.linspace(-2, 2, 10000)

    arg_grid = np.subtract.outer(y, x).T

    rates = ((arg_grid >= -1) & (arg_grid <= 1))
    rates = rates / rates.sum(axis=1, keepdims=True)

    res = optimize(rates, max_iter=int(1e5), eps=1e-2)
    print('done')