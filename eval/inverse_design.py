import os
import torch
import numpy as np
from PIL import Image
from utils import Evaluation
from omegaconf import OmegaConf


def load_amp_and_phase_mask(path):
    amp = Image.open(os.path.join(path, 'amp.png')).convert('L')
    phase = Image.open(os.path.join(path, 'phase.png')).convert('L')

    amp = np.flip(np.asarray(amp) / 255.0, axis=0)
    phase = np.flip(np.asarray(phase) / 255.0, axis=0)
    phase = phase * 2 * np.pi - np.pi

    res = amp * np.exp(1j * phase)

    return np.real(res), np.imag(res)


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


if __name__ == '__main__':
    """ Load configuration. """
    cfg = OmegaConf.load('../configs/meta-atom.yaml')
    cfg = OmegaConf.create(OmegaConf.to_yaml(cfg, resolve=True))

    """ Load model """
    eval = Evaluation('../models/inn-meta-atom-resnet-xy.pt', cfg)

    """ Inverse Design """
    """
    output, record = [], {}
    real_d, imag_d = load_amp_and_phase_mask('aperture/D')
    real_l, imag_l = load_amp_and_phase_mask('aperture/L')

    for rd, id, rl, il in zip(real_d.flatten(), imag_d.flatten(), real_l.flatten(), imag_l.flatten()):
        if (rd, id, rl, il) not in record:
            # Sample posterior
            params = eval.sample_posterior([rd, id, rl, il], n=128)
            # Re-simulate and sorting
            pred_targets = eval.forward_prediction(params)
            distance = np.linalg.norm(pred_targets - np.array([rd, id, rl, il]), axis=1)
            idx = np.argsort(distance)
            record[(rd, id, rl, il)] = params[idx[0]] * 72.1688 + 175

        output.append(record[(rd, id, rl, il)])

    output = np.array(output)

    # Save result
    if not os.path.isdir('res'):
        os.makedirs('res')
    np.savetxt('res/dl_3.csv', output, delimiter=',', fmt='%.2f')
    """

    params = np.loadtxt('res/dl_2.csv', delimiter=',')
    params = (params - 175) / 72.1688
    res = eval.forward_prediction(params)
    res_x, res_y = res[:, 0] + 1j * res[:, 1], res[:, 2] + 1j * res[:, 3]
    res_x = res_x.reshape(50, 50)
    res_y = res_y.reshape(50, 50)
    np.save('res/res_x2.npy', res_x)
    np.save('res/res_y2.npy', res_y)

