"""the main use of functions in this file is to convert results from scanning to
grids with capacity
"""

from analysis.information import CIDOChannel
from joblib.parallel import Parallel, delayed
import numpy as np

def res_to_info(res, per_spike=False, **kwargs):
    channel = CIDOChannel(res)
    info_res = channel.max_information(**kwargs)

    if per_spike:
        mean_out = np.arange(len(info_res['out_pdf'])).dot(info_res['out_pdf'])
        return info_res['fun'] / mean_out
    else:
        return info_res['fun']

def compute_information_grids(scan_results, max_spikes, n_jobs, **kwargs):
    info_results = Parallel(n_jobs=n_jobs, temp_folder='/tmp')(delayed(res_to_info)(
        res=res,
        method='slsqp',
        max_spikes=max_spikes,
        **kwargs) for res in scan_results.values()
    )

    param_list = []
    for param in scan_results.keys():
        alpha1, alpha2 = eval(param)
        param_list.append((alpha1, alpha2))

    return {param: res for param, res in zip(param_list, info_results)}

def info_grid_to_img(info_grid):
    x = []
    y = []
    z = []

    for params, capacity in info_grid.items():
        x.append(params[0])
        y.append(params[1])
        z.append(capacity)

    x_unq = np.unique(x)
    y_unq = np.unique(y)

    img = np.empty((len(y_unq), len(x_unq)))

    for i, alpha1 in enumerate(x_unq):
        for j, alpha2 in enumerate(y_unq):
            img[j, i] = info_grid[(alpha1, alpha2)]

    return img

if __name__ == '__main__':
    import json
    import matplotlib.pyplot as plt

    with open('outputs/MAT2scan.json') as f:
        scan_results = json.load(f)

    f, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
    imgs = []

    for ix, max_spikes in enumerate([100, 50, 20, 8]):
        information_grid = compute_information_grids(scan_results, max_spikes=max_spikes, n_jobs=4, per_spike=True)

        img = info_grid_to_img(information_grid)
        imgs.append(img)

        i, j = np.unravel_index(ix, axes.shape)
        ax = axes[i, j]
        ax.set_title('max avg sp. cnt: {}'.format(max_spikes))
        ax.set_xlabel('alpha1')
        ax.set_ylabel('alpha2')
        ax.imshow(img, cmap='Blues', origin='lower')

        print(max_spikes, 'done')

    x = []
    y = []
    z = []

    for params, capacity in information_grid.items():
        x.append(params[0])
        y.append(params[1])
        z.append(capacity)

    x_unq = np.unique(x)
    y_unq = np.unique(y)

    ticks = np.arange(0, img.shape[0], 5)

    for ix in range(4):
        i, j = np.unravel_index(ix, axes.shape)
        ax = axes[i, j]

        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        ax.set_xticklabels([f'{x_unq[t]:.0f}' for t in ticks])
        ax.set_yticklabels([f'{y_unq[t]:.0f}' for t in ticks])

    f.tight_layout()
    plt.savefig('img/heatmaps_per_spike.png')