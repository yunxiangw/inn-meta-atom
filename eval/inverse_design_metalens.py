import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from utils import Evaluation
from omegaconf import OmegaConf


""" Matplotlib Parameters """
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': 'Time',
    'font.size': 26,
})


def build_forward_inputs(x):
    """ Build input for forward prediction. """
    x_pad = cfg['data']['input_dim'] - cfg['data']['x_dim']
    x = torch.cat((torch.Tensor(x), torch.zeros(x.shape[0], x_pad)), dim=1) if x_pad else x
    return x


def build_backward_inputs(y):
    """ Build input for backward retrival. """
    zy_pad = cfg['data']['input_dim'] - cfg['data']['y_dim'] - cfg['data']['z_dim']

    y = torch.cat((torch.zeros(y.shape[0], zy_pad), torch.Tensor(y)), dim=1) if zy_pad else y
    y = torch.cat((torch.randn(y.shape[0], cfg['data']['z_dim']), y), dim=1)
    return y


def plot_exz(efield):
    fig, ax = plt.subplots(layout='constrained', dpi=300)
    ax.imshow(efield, cmap='hot')
    ax.set_xticks((0, 400/3, 800/3, 400))
    ax.set_xticklabels(('$0$', '$50$', '$100$', '$150$'))
    ax.set_yticks((0, 200))
    ax.set_yticklabels(('$40$', '$0$'))
    ax.set_aspect(2/3.75)
    ax.set_xlabel('$z (\mu m)$')
    ax.set_ylabel('$x (\mu m)$')
    plt.savefig('2xz_sim.tif')


def plot_exy(efield):
    fig, ax = plt.subplots(layout='constrained', dpi=300)
    ax.imshow(efield, cmap='hot')
    ax.set_aspect(1)
    plt.axis('off')
    plt.savefig('1xy.tif')


if __name__ == '__main__':

    # Load configuration.
    cfg = OmegaConf.load('../configs/rectangle.yaml')
    cfg = OmegaConf.create(OmegaConf.to_yaml(cfg, resolve=True))

    # Load model
    eval = Evaluation('../models/inn-rectangle.pt', cfg)
    """
    # Load phase mask 
    original_phase = np.load('res/phase_profile.npy')
    # 0.276796 -- 6.8377743
    original_phase = original_phase + np.pi + 0.276796
    original_phase = np.round(original_phase, 6)
    phase_map = original_phase.flatten()

    # Inverse retrieval
    all_phases = np.sort(np.unique(phase_map))
    all_params = []
    for p in all_phases:
        params = eval.sample_posterior(target=p, n=20)
        all_params.append(params[0].tolist())
    phase_to_param = {p: [i * 92.3760 + 200 for i in param] for p, param in zip(all_phases, all_params)}

    # Forward prediction
    rebuild_phases = eval.forward_prediction(all_params)
    phase_to_rebuild = {p: rp for p, rp in zip(all_phases, rebuild_phases)}

    # Save forward prediction result
    x, y = np.linspace(-19.8, 19.8, 100), np.linspace(-19.8, 19.8, 100)
    xx, yy = np.meshgrid(x, y)

    output = []
    for cx, cy, p in zip(xx.flatten(), yy.flatten(), phase_map):
        if (cx ** 2 + cy ** 2) < 20 ** 2:
            output.append([np.round(cx, 1), np.round(cy, 1)] + phase_to_param[p])

    res = np.array(output)
    np.savetxt('params_3.txt', res, delimiter=',', fmt='%.4f')

    # Save rebuild phase profile
    rebuild_phase = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            rebuild_phase[i, j] = phase_to_rebuild[original_phase[i, j]] - np.pi - 0.276796
    np.save('rebuild_phase_profile_3.npy', rebuild_phase)
    """

    # e = loadmat('res/2xz.mat')['E2_xz']
    # plot_exz(e)

    phase_profile = np.load('rebuild_phase_profile_1.npy')
    x, y = np.linspace(-19.8, 19.8, 100), np.linspace(-19.8, 19.8, 100)
    xx, yy = np.meshgrid(x, y)
    mask = np.where((xx ** 2 + yy ** 2) < 20 ** 2, 1, np.zeros_like(xx))
    phase_profile *= mask

    fig, ax = plt.subplots(layout='constrained', dpi=300)
    ax.imshow(phase_profile, cmap='gray')
    plt.axis('off')
    plt.savefig('rebuild_phase_profile.tif')
