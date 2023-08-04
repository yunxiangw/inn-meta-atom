import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import Evaluation
from scipy.io import loadmat, savemat
from omegaconf import OmegaConf
from pandas.plotting import parallel_coordinates

""" Matplotlib Parameters """
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': 'Time',
    'font.size': 28,
})


def abc(y, eps=0.05):
    params = loadmat('../data/meta-atom/params.mat')['params'][:55000, :]
    targets = loadmat('../data/meta-atom/real_imag.mat')['output']
    # targets = targets[:, :2]

    params_list = []
    targets_list = []

    for p, t in zip(params, targets):
        if np.linalg.norm(t - y) <= eps:
            params_list.append(p)
            targets_list.append(t)
    return np.array(params_list)


def plot_abc(targets, eps):
    res = abc(targets, eps=eps)
    label = ['W1', 'L1', 'W2', 'L2']
    df = pd.DataFrame(res, columns=label)
    df.loc[:, 'class'] = 0

    fig, ax = plt.subplots(layout='constrained', dpi=300)
    ax.set_ylim((50, 300))
    ax.set_yticks((100, 200, 300))

    parallel_coordinates(df, class_column='class', color=('#A6559D',), ax=ax)
    ax.get_legend().remove()

    plt.show()


def plot_posterior(eval, targets):
    # Inverse retrieval
    label = ['W1', 'L1', 'W2', 'L2']

    params = eval.sample_posterior(targets, n=200)
    params = params * 72.1688 + 175

    df = pd.DataFrame(params, columns=label)
    df.loc[:, 'class'] = 0

    fig, ax = plt.subplots(layout='constrained', dpi=300)
    ax.set_ylim((50, 300))
    ax.set_yticks((100, 200, 300))

    parallel_coordinates(df, class_column='class', color=('#A6559D',), ax=ax)
    ax.get_legend().remove()

    plt.show()


if __name__ == '__main__':

    """ Load configuration. """
    cfg = OmegaConf.load('../configs/meta-atom.yaml')
    cfg = OmegaConf.create(OmegaConf.to_yaml(cfg, resolve=True))

    eval = Evaluation('../models/inn-meta-atom-2.pt', cfg)

    """
    # Forward prediction
    test_params = np.load('../data/meta-atom/test/params.npy')
    test_params = (test_params - 175) / 72.1688
    test_targets = np.load('../data/meta-atom/test/targets.npy')
    pred_targets = eval.forward_prediction(test_params)

    error = np.sum((pred_targets[:, :2] - test_targets[:, :2]) ** 2, axis=1)

    fig, ax = plt.subplots(layout='constrained', dpi=300)
    ax.hist(error, bins=20, range=(0, 0.02), color='#ea5514', edgecolor='black', linewidth=1)

    ax.set_xlim((0, 0.02))
    ax.set_xticks((0, 0.01, 0.02))
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax.set_xlabel('MSE')

    ax.set_ylim((0, 800))
    ax.set_yticks((0, 400, 800))
    ax.set_ylabel('Counts')

    plt.savefig('x_error.tif')
    """

    """
    dim = [dict(range=[50, 350],
                tickvals=[50, 100, 150, 200, 250, 300, 350],
                label=label[i],
                values=res[:, i]) for i in range(4)]
    fig = go.Figure(data=go.Parcoords(dimensions=dim))
    fig.show()
    """

    """
    # Re-simulate and sorting
    pred_targets = eval.forward_prediction(params)
    distance = np.linalg.norm(pred_targets - np.array(targets), axis=1)

    idx = np.argsort(distance)
    params, pred_targets = params[idx], pred_targets[idx, :]
    res = np.concatenate((params, pred_targets), axis=1)

    fig, ax = plt.subplots(layout='constrained', dpi=300)
    ax.hist(distance, bins=10, range=(0, 1), color='#ea5514', edgecolor='black', linewidth=1)

    ax.set_xlim((0, 1))
    ax.set_xticks((0, 0.2, 0.4, 0.6, 0.8, 1))
    ax.set_xlabel('MSE')

    ax.set_ylim((0, 30))
    ax.set_yticks((0, 10, 20, 30))
    ax.set_ylabel('Counts')

    plt.show()

    np.save('1.npy', res)
    """


    targets = [0.5, 0, 0, -0.5]
    # plot_abc(targets, eps=0.15)

    # Inverse retrieval
    params = eval.sample_posterior(targets, n=100)

    # Re-simulate and sorting
    pred_targets = eval.forward_prediction(params)
    distance = np.linalg.norm(pred_targets - np.array(targets), axis=1)

    idx = np.argsort(distance)
    params, pred_targets = params[idx], pred_targets[idx, :]
    savemat('params.mat', {'params': params})
    savemat('prediction.mat', {'prediction': pred_targets})

