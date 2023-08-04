import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import torch
from omegaconf import OmegaConf
from networks import build_inn_model

""" Matplotlib Parameters """
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': 'Time',
    'font.size': 28,
})

""" Load trained INN model """
cfg = OmegaConf.load('../configs/rectangle.yaml')
cfg = OmegaConf.create(OmegaConf.to_yaml(cfg, resolve=True))


def plot_phase(fig, ax, phase, file_name=None):
    im = ax.imshow(
        phase,
        origin='lower',
        cmap='RdBu',
        vmin=0,
        vmax=2 * np.pi,
        interpolation=None,
        extent=[50, 350, 50, 350]
    )

    ax.set_xlim([50, 350])
    ax.set_ylim([50, 350])
    ax.set_xticks([50, 150, 250, 350])
    ax.set_yticks([50, 150, 250, 350])
    ax.set_xlabel('Width (nm)')
    ax.set_ylabel('Length (nm)')
    ax.get_yaxis().set_visible(False)

    cbar = fig.colorbar(im, ax=ax, pad=0, orientation='vertical', ticks=[0, np.pi, 2 * np.pi])
    cbar.ax.set_yticklabels(['$0$', '$\pi$', '$2\pi$'])
    if file_name:
        fig.savefig(f'{file_name}')
    return fig, ax


def forward_prediction(x):
    x_pad = cfg['data']['input_dim'] - cfg['data']['x_dim']
    x = torch.cat((x, torch.zeros(x.shape[0], x_pad)), dim=1)
    x = x.to(cfg['device'])

    with torch.no_grad():
        y, _ = model(x)
        y = y[:, -cfg['data']['y_dim']:]
    return y.detach().cpu().numpy()


def sample_posterior(target, n=10):
    x_dim, y_dim, z_dim = cfg['data']['x_dim'], cfg['data']['y_dim'], cfg['data']['z_dim']
    zy_pad = cfg['data']['input_dim'] - y_dim - z_dim

    rev_inputs = torch.cat((torch.randn(n, z_dim), torch.zeros(n, zy_pad + y_dim)), dim=1)
    rev_inputs[:, -y_dim:] = target
    rev_inputs = rev_inputs.to(cfg['device'])

    with torch.no_grad():
        x_pred, _ = model(rev_inputs, rev=True)
        x_pred = x_pred[:, :x_dim].data.cpu().numpy()
        x_pred = x_pred * 92.3760 + 200  # Map back
        mask = np.all(x_pred >= 50, axis=1) & np.all(x_pred <= 350, axis=1)
        x_pred = x_pred[mask, :]

    return x_pred


def plot_ground_truth():
    y = np.load('../data/rectangle/grid/targets.npy')
    y = y.reshape(101, 101, order='F')
    fig, ax = plt.subplots(layout='constrained', dpi=300)
    plot_phase(fig, ax, y, 'ground_truth.tif')


def plot_forward_prediction():
    params_grid = torch.Tensor(np.load('../data/rectangle/grid/params.npy'))
    params_grid = (params_grid - 200) / 92.3760

    targets_pred = forward_prediction(params_grid)
    l = int(np.sqrt(len(targets_pred)))
    targets_pred = targets_pred.reshape(l, l, order='F')

    fig, ax = plt.subplots(layout='constrained', dpi=300)
    plot_phase(fig, ax, targets_pred, 'forward_prediction.tif')


def plot_posterior(targets, n_samples=10, savefig=False):
    fig, ax = plt.subplots(layout='constrained', dpi=300)

    # Plot the ground truth and the contour
    params_grid = np.load('../data/rectangle/grid/params.npy')
    targets_grid = np.load('../data/rectangle/grid/targets.npy')

    x_grid = params_grid[:, 0].reshape(101, 101, order='F')
    y_grid = params_grid[:, 1].reshape(101, 101, order='F')
    targets_grid = targets_grid.reshape(101, 101, order='F')

    fig, ax = plot_phase(fig, ax, targets_grid)

    ax.contour(x_grid, y_grid, targets_grid,
               levels=targets,
               colors='black',
               linewidths=3,
               linestyles='dashed')

    # Plot prediction
    strs = [r'${:.1f}\pi$'.format(i / np.pi) for i in targets]

    for i, t in enumerate(targets):
        x_pred = sample_posterior(t, n_samples)
        ax.scatter(x_pred[:, 0], x_pred[:, 1],
                   c=[i] * x_pred.shape[0],
                   vmin=0, vmax=9, cmap='tab10', marker='p', s=100, label=strs[i])

    ax.legend(loc='lower left', fontsize=26, borderpad=0.1, borderaxespad=0.1, labelspacing=0.1, handlelength=1,
              handletextpad=0.2)
    if savefig:
        fig.savefig('4.tif')


def abc(y, eps=0.05):
    params = np.load('../data/rectangle/train/params.npy')
    targets = np.load('../data/rectangle/train/targets.npy')

    params_list = []
    targets_list = []
    for p, t in zip(params, targets):
        if np.linalg.norm(t - y) <= eps:
            params_list.append(p)
            targets_list.append(t)
    return np.array(params_list)


if __name__ == '__main__':
    # Load trained INN model
    model = build_inn_model(**cfg['model'])
    checkpoint_dicts = torch.load('../models/inn-rectangleepoch99.pt')
    model.load_state_dict(checkpoint_dicts['net'])
    model.to(cfg['device'])

    # Load grid data for plotting
    # params_grid = torch.Tensor(np.load('../data/rectangle/grid/params.npy'))
    # targets_grid = torch.Tensor(np.load('../data/rectangle/grid/targets.npy'))

    # Plot ground truth
    # plot_ground_truth()

    # Plot forward prediction
    # plot_forward_prediction()

    # Plot posterior
    targets = np.array([0.5, 1.0, 1.5, 2.0]) * np.pi
    plot_posterior(targets, n_samples=20, savefig=True)

