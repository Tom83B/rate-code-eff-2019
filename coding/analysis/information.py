import numpy as np
import pandas as pd
from scipy.optimize import minimize, bisect
from scipy.stats import poisson
from scipy.interpolate import RectBivariateSpline
from optimization.blahut_arimoto import optimize as ba_optimize
from optimization.blahut_arimoto import alphabet_opt as chang
from optimization.slsqp import optimize as slsqp_opt
from optimization.jimbo_kunisawa import optimize as jimbo_opt
from analysis.plotting import stimulus_response_hm, plot_discrete_distribution, stimulus_response_line, plotter
import json
import itertools
import matplotlib.pyplot as plt


def relative_entropy(pdf1, pdf2):
    return (pdf1 * np.nan_to_num(np.log2(pdf1 / pdf2))).sum()

def channel_discretization(pdf_func, input_range, m, eps=1e-5):
    x_arr = [input_range[0]]
    z_arr = []

    for j in itertools.count():
        pdf1 = pdf_func(x_arr[-1])
        opt_func = lambda y: relative_entropy(pdf1, pdf_func(y)) - eps

        if opt_func(x_arr[-1]) * opt_func(input_range[1]) < 0:
            z_arr.append(bisect(opt_func, x_arr[-1], input_range[1]))
            pdf2 = pdf_func(z_arr[-1])

            opt_func = lambda y: relative_entropy(pdf_func(y), pdf2) - eps

            if opt_func(z_arr[-1]) * opt_func(input_range[1]) < 0:
                x_arr.append(bisect(opt_func, z_arr[-1], input_range[1]))
            elif j < m:
                return channel_discretization(pdf_func, input_range, m, eps / 2)
            else:
                x_arr[-1] = input_range[1]
                return (x_arr, z_arr)
        elif j < m:
            return channel_discretization(pdf_func, input_range, m, eps / 2)
        else:
            z_arr[-1] = input_range[1]
            return (x_arr, z_arr)

        print(x_arr[-1], z_arr[-1])


from analysis.plotting import plotter
from scipy.interpolate import UnivariateSpline

class AdaptationChannel:
    def __init__(self, rate_data):
        self.rate_data = rate_data
        self.info_res = {
            'jimbo': None,
            'blahut': None
        }
    
    def obtain_efficiency(self, method='jimbo', n_iter=5, **kwargs):
        if method == 'jimbo':
            opt_fun = jimbo_opt
        elif method == 'blahut':
            opt_fun = ba_optimize
        
        transitions = self.rate_data
        pdf = np.ones(transitions.shape[0])
        pdf = pdf / pdf.sum()

        max_rate = transitions.map(lambda x: x.shape[1]).max()
        
        for i in range(n_iter):
            rates = np.zeros((len(pdf), max_rate + 1))

            for p, (ix, tmp_rates) in zip(pdf, transitions.iteritems()):
                stim_max_rate = tmp_rates.shape[1]
                rates += p * np.pad(tmp_rates, [(0,0), (0, max_rate - stim_max_rate + 1)], mode='constant')

            info_res = opt_fun(rates, **kwargs)
            info_res['rates'] = rates
            pdf = info_res['pdf']
        
        self.info_res[method] = info_res
        
        return info_res
    
    def cost_capacity(self, min_rate=9, resolution_points=10, **kwargs):
        to_be_df = {}
        
        for s in np.linspace(0, 1, 6):
            print(s)
            info_res = self.obtain_efficiency(method='blahut', s=s)
            to_be_df[s] = {'W': info_res['expense'],
                           'C': info_res['fun'],
                           'C/W': info_res['fun'] / info_res['expense']}
            if info_res['expense'] < min_rate:
                break
        
        temp_df = pd.DataFrame(to_be_df).T.sort_values(by='W')
        min_expense = temp_df['W'].min()
        max_expense = temp_df['W'].max()
        
        expense_s_func = UnivariateSpline(x=temp_df['W'], y=temp_df.index, s=0, k=1)
        
        new_s_arr = expense_s_func(np.linspace(min_expense, max_expense, resolution_points))
        print(new_s_arr)
        
        for s in new_s_arr[1:][::-1]:
            print(s)
            info_res = self.obtain_efficiency(method='blahut', s=s)
            to_be_df[s] = {'W': info_res['expense'],
                           'C': info_res['fun'],
                           'C/W': info_res['fun'] / info_res['expense']}
            if info_res['expense'] < min_rate:
                break
        
        return pd.DataFrame(to_be_df).T
    
    @plotter
    def plot(self, method='jimbo', *, ax=None, cmap='Blues', **kwargs):
        if self.info_res[method] is None:
            _ = self.obtain_efficiency(method=method)
        
        rates = self.info_res[method]['rates']
        
        ax.imshow(rates.T, origin='lower', aspect='auto', cmap=cmap, **kwargs)
        
        outs = np.arange(rates.shape[1])
        mu = rates.dot(outs)
        std = np.sqrt(np.clip(rates.dot(outs * outs) - mu ** 2, 0, np.inf))
        
        ax.plot(mu, c='white')
        ax.plot(mu - std, c='white', linestyle='dashed')
        ax.plot(mu + std, c='white', linestyle='dashed')


class InformationChannel:
    """Base class for all information channels"""
    def __init__(self, experiment_data=None, **interpolation_kwargs):
        """Accepts dict in form:
        {
            stimulus_0: [count_0, count_1, ..., coutn_n_runs],
            stimulus_1: ...
            ...
            stimulus_N: ...
        }
        where stimulus_0 ... stimulus_N are all the different stimuli applied in the experiment
        count_0, ... count_n_runs are the numbers of recorded spikes in each of n_run experiments for given stimulus

        OR accepts path to json file with the same structure

        :param experiment_data: either dict or path to json file containing experiment results
        """
        if type(experiment_data) == dict:
            self.intensities = np.array([float(x) for x in experiment_data.keys()])
            self.rates = self._counts_to_rates(experiment_data)
        elif type(experiment_data) == pd.Series:
            self.__init__(experiment_data.to_dict(), **interpolation_kwargs)
        elif type(experiment_data) == str:
            self.load_json(experiment_data)
        elif experiment_data is not None:
            raise Warning('experiment_data has to be either dict or path to json file')
        else:
            pass

    def load_json(self, filename):
        with open(filename, 'r') as infile:
            counts = json.load(infile)
        self.intensities = np.array([float(x) for x in counts.keys()])
        self.rates = self._counts_to_rates(counts)

    @staticmethod
    def _counts_to_rates(spike_counts, max_rate=None):
        """Converts experiment results given as dict to grid with conditional rates f(r|theta).

        :param spike_counts: dictionary, see __init__
        :return: NDArray of shape number_of_stimuli x number_of_possible_outcomes: (obracene)
            [[f(r_0|theta_0), f(r_0|theta_1),   ...  ],
             [f(r_1|theta_0), ...                    ],
             ...
             [f(r_max|theta_0), ..., f(r_max|theta_N)]]
        """
        inputs = list(spike_counts.keys())
        inputs.sort(key=float)
        rates = []
        if max_rate is None:
            max_rate = np.max([np.max(l) for l in spike_counts.values()])
        bins = list(range(0, max_rate + 2))

        for freq in inputs:
            rates.append(np.histogram(spike_counts[freq], bins=bins, normed=True)[0])

        return np.array(rates)


class CIDOChannel(InformationChannel):
    """class for continuous-input-discrete-output information channels"""
    def __init__(self, experiment_data=None, interpolation_kwargs=None):
        super(CIDOChannel, self).__init__(experiment_data)

        if experiment_data is not None:
            if interpolation_kwargs is None:
                interpolation_kwargs = {
                    'kx': 1,
                    'ky': 1
                }

            self.interp_func = self._get_interpolating_function()
        else:
            self.interp_func = None

    def output_pdf(self, x):
        if self.interp_func is None:
            raise Exception('channel has no data to interpolate from')
        else:
            return self.interp_func(x)

    def _get_interpolating_function(self):
        y = np.arange(self.rates.shape[1])
        func = RectBivariateSpline(self.intensities, y, self.rates, kx=1, ky=1)

        def pf(x):
            # print(','.join(np.atleast_1d(x).astype(str)))
            # print()
            ret = func(np.clip(x, self.intensities[0], self.intensities[-1]),y)
            if np.isscalar(x):
                return ret[0]
            else:
                return ret
        return pf
        # return lambda x: func(x, y)

    def _adjust_grid(self, n_intensities):
        intensities_new = np.linspace(float(min(self.intensities)), float(max(self.intensities)), n_intensities)
        y = np.arange(self.rates.shape[1])
        func = RectBivariateSpline(self.intensities, y, self.rates, kx=1, ky=1)
        self.rates = func(intensities_new, y)
        self.intensities = intensities_new

    def _generate_alphabet(self):
        return channel_discretization(self.interp_func, (self.intensities.min(), self.intensities.max()),
                                              m=self.rates.shape[1])

    def max_information(self, method='blahut_arimoto', generate_alphabet=False, **kwargs):
        """
        Finds the capacity of the information channel
        :param method: algorithm to use. choose from:
            * blahut_arimoto
            * chang
            * slqp - constrained optimization method implemented in SciPy
            * jimbo - the Jimbo-Kunisawa algorithm for relative capacity computation
        :param generate_alphabet: not implemented
        :param kwargs: keyword arguments for the optimization algorithm
        :return: the optimization result
        """
        rates = self.rates
        if generate_alphabet is True:
            raise NotImplementedError('generate_alphabet option is not well implemented')
            alphabet = channel_discretization(self.interp_func, (self.intensities.min(), self.intensities.max()),
                                              m=self.rates.shape[1])

        if method == 'blahut_arimoto':
            return ba_optimize(self.rates, **kwargs)
        elif method == 'chang':
            return chang(self.interp_func, self.intensities.min(), self.intensities.max(), **kwargs)
        elif method == 'slsqp':
            return slsqp_opt(self.rates, **kwargs)
        elif method == 'cutting_plane':
            return cp_optimize(self.rates, **kwargs)
        elif method == 'jimbo':
            return jimbo_opt(self.rates, **kwargs)
        else:
            raise ValueError('unknown optimization method')

    def plot(self, plot_type='heatmap', **kwargs):
        if plot_type == 'heatmap':
            stimulus_response_hm(self.intensities, self.rates, **kwargs)
        elif plot_type == 'lines':
            stimulus_response_line(self.intensities, self.rates, **kwargs)

    def plot_opt_pdf(self, method='blahut_arimoto', eps=1e-5, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        res = self.max_information(method=method, eps=eps)
        plot_discrete_distribution(
            intensities=self.intensities,
            pdf=res['pdf'],
            ax=ax,
            **kwargs
        )


class PoissonChannel:
    def __init__(self, tuning_curve, min_intensity, max_intensity):
        self.tuning_curve = np.vectorize(tuning_curve)
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity
        k = 20

        while poisson.cdf(k, mu=max_intensity) < 1 - 1e-4:
            k += 1

        self.out_vec = np.arange(k)

        def out_func(intensity):
            mu = self.tuning_curve(intensity)

            if len(np.shape(mu)) == 0:
                pmf = poisson.pmf(k=self.out_vec, mu=mu)
                return pmf / pmf.sum()
            else:
                out = []
                for m in mu:
                    pmf = poisson.pmf(k=self.out_vec, mu=m)
                    out.append(pmf / pmf.sum())
                return np.array(out)

        self.out_func = out_func

    def max_information(self, method='chang', **kwargs):
        if method == 'chang':
            return chang(self.out_func, min_intensity=self.min_intensity,
                         max_intensity=self.max_intensity, **kwargs)
        elif method == 'discr':
            grid = np.linspace(self.min_intensity, self.max_intensity, 100)
            sr_grid = self.out_func(grid)
            return ba_optimize(sr_grid, **kwargs)


class DIDOChannel(InformationChannel):
    """class for discrete-input-discrete-output information channels"""
    def __init__(self, experiment_data=None):
        super(DIDOChannel, self).__init__(experiment_data)

    def _adjust_grid(self, n_intensities):
        intensities_new = np.linspace(float(min(self.intensities)), float(max(self.intensities)), n_intensities)
        y = np.arange(self.rates.shape[1])
        func = RectBivariateSpline(self.intensities, y, self.rates, kx=1, ky=1)
        self.rates = func(intensities_new, y)
        self.intensities = intensities_new

    def _rate_prior(self, stimulus_pdf):
        return stimulus_pdf.dot(self.rates)

    def _sr_information_grid(self, stimulus_pdf):
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.nan_to_num(np.log2(self.rates / self._rate_prior(stimulus_pdf)))

    def information(self, stimulus_pdf):
        return stimulus_pdf.dot(self._sr_information_grid(stimulus_pdf) * self.rates).sum()

    def optimize(self, **kwargs):
        return ba_optimize(self.rates, **kwargs)


if __name__ == '__main__':
    from analysis.plotting import stimulus_response_hm
    import matplotlib.pyplot as plt

    grid = CIDOChannel('outputs/mat2_bivoj.json')
    # f, (ax0, ax1) = plt.subplots(ncols=2, figsize=(15,9))
    # stimulus_response_hm(grid.intensities, grid.rates, ax0)

    f, ax = plt.subplots(nrows=10, figsize=(15,9), sharex=True, sharey=True)
    f.subplots_adjust(hspace=0.1)

    font = dict(
        size=16
    )

    for i in range(10):
        ax[i].step(np.arange(0, grid.rates.shape[1], 1), grid.rates[i * 70])
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['left'].set_visible(False)
        ax[i].yaxis.set_ticks_position('none')
        ax[i].yaxis.set_ticklabels([])
        ax[i].set_xticks([0,10,20,30,40,50])
        ax[i].set_xlim((0,50))
        khz_int = grid.intensities[i * 70] / 1000
        ax[i].text(x=45, y=0.1, s=f'{khz_int:.1f} kHz', fontdict=font)
        if i!= 10:
            ax[i].spines['bottom'].set_visible(False)

    # ax1.step(np.arange(0, grid.rates.shape[1], 1), grid.rates[699])
    # ax1.step(np.arange(0, grid.rates.shape[1], 1), grid.rates[599])
    # ax1.step(np.arange(0, grid.rates.shape[1], 1), grid.rates[499])
    # ax1.step(np.arange(0, grid.rates.shape[1], 1), grid.rates[399])
    # ax1.step(np.arange(0, grid.rates.shape[1], 1), grid.rates[299])
    # ax1.step(np.arange(0, grid.rates.shape[1], 1), grid.rates[199])
    # ax1.step(np.arange(0, grid.rates.shape[1], 1), grid.rates[99])
    # ax1.step(np.arange(0, grid.rates.shape[1], 1), grid.rates[0])
    plt.savefig('img/rates.png')