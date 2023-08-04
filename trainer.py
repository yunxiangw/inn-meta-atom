import os
import torch
import numpy as np

from os.path import isdir, join
from collections import defaultdict


def generate_inputs(
        x, y,
        input_dim, x_dim, y_dim, z_dim,
        y_noise, z_noise, pad_noise
):
    x_pad, zy_pad = input_dim - x_dim, input_dim - y_dim - z_dim
    batch_size = x.shape[0]
    if y_noise:
        y += y_noise * torch.randn(y.shape)
    if x_pad:
        x = torch.cat((x, pad_noise * torch.randn(batch_size, x_pad)), dim=1)
    if zy_pad:
        y = torch.cat((pad_noise * torch.randn(batch_size, zy_pad), y), dim=1)
    y = torch.cat((torch.randn(batch_size, z_dim), y), dim=1)

    return x, y


def generate_perturbed_targets(
        pred_targets, device,
        input_dim, x_dim, y_dim, z_dim,
        y_noise, z_noise, pad_noise
):
    zy_pad = input_dim - y_dim - z_dim
    batch_size = pred_targets.shape[0]

    pred_z, pad, pred_y = pred_targets[:, :z_dim], pred_targets[:, z_dim: -y_dim], pred_targets[:, -y_dim:]
    perturbed_targets = pred_z + z_noise * torch.randn(batch_size, z_dim).to(device)
    if zy_pad:
        pad = pad + pad_noise * torch.randn(batch_size, zy_pad).to(device)
        perturbed_targets = torch.cat((perturbed_targets, pad), dim=1)
    perturbed_targets = torch.cat((perturbed_targets, pred_y + y_noise * torch.randn(batch_size, y_dim).to(device)), dim=1)

    return perturbed_targets


def sample_posterior(model, target, n, device,
                     input_dim, x_dim, y_dim, z_dim,
                     y_noise, z_noise, pad_noise):
    zy_pad = input_dim - y_dim - z_dim

    rev_inputs = torch.cat((torch.randn(n, z_dim + zy_pad), torch.zeros(n, y_dim)), dim=1)
    rev_inputs[:, z_dim:-y_dim] *= pad_noise
    rev_inputs[:, -y_dim:] = torch.Tensor(target)
    rev_inputs = rev_inputs.to(device)

    with torch.no_grad():
        outputs, _ = model(rev_inputs, rev=True)
        outputs = outputs.detach().cpu().numpy()
        # Rectangle: mean: 200, std: 92.3760
        # Meta-atom: mean: 175, std: 72.1688
        outputs = outputs[:, :x_dim] * 72.1688 + 175
    return outputs


def train_epoch(
        cfg_data,
        epoch,
        model,
        criterion,
        train_loader,
        optimizer,
        mode='init'
):

    """ Training procedure. """
    # Switch to train mode and clear the gradient.
    model.train()
    optimizer.zero_grad()

    loss_history = defaultdict(list)
    pred_params_list, pred_targets_list = [], []

    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'

    for params, targets in train_loader:
        # Pad zeros for both params and targets
        params, targets = generate_inputs(params, targets, **cfg_data)
        params, targets = params.to(device), targets.to(device)

        # Generate predictions
        pred_targets, _ = model(params)
        pred_params, _ = model(targets, rev=True)

        perturbed_targets = generate_perturbed_targets(pred_targets.detach(), device, **cfg_data)  # Need to detach
        reconstruct_params, _ = model(perturbed_targets, rev=True)

        pred_targets_list.append(pred_targets.detach().cpu().numpy())
        pred_params_list.append(pred_params.detach().cpu().numpy())

        # Calculate loss
        batch_loss = criterion(params, targets, pred_params, pred_targets, reconstruct_params, mode)
        for name, val in batch_loss.items():
            loss_history[name].append(val.item())

        loss = sum(l for l in batch_loss.values())

        # Generate the gradient and accumulate
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    loss_history = {name: np.mean(val) for name, val in loss_history.items()}
    return loss_history


def test_epoch(
        cfg_data,
        epoch,
        model,
        criterion,
        test_loader,
        plotter,
        mode
):
    """ Test procedure. """
    # Create the directories.
    if not isdir('test'):
        os.makedirs('test')

    # Switch to evaluation mode.
    model.eval()

    loss_history = defaultdict(list)
    pred_params_list, pred_targets_list = [], []

    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'

    with torch.no_grad():
        # Evaluate the model
        pos = sample_posterior(model, [0.7, -0.7, 0.7, -0.7], 256, device, **cfg_data)

        for params, targets in test_loader:
            # Pad zeros for both params and targets
            params, targets = generate_inputs(params, targets, **cfg_data)
            params, targets = params.to(device), targets.to(device)

            # Generate predictions
            pred_targets, _ = model(params)
            pred_params, _ = model(targets, rev=True)

            perturbed_targets = generate_perturbed_targets(pred_targets.detach(), device, **cfg_data)  # Need to detach
            params_reconstruct, _ = model(perturbed_targets, rev=True)

            pred_params_list.append(pred_params.detach().cpu().numpy())
            pred_targets_list.append(pred_targets.detach().cpu().numpy())

            # Calculate loss
            batch_loss = criterion(params, targets, pred_params, pred_targets, params_reconstruct, mode=mode)
            for name, val in batch_loss.items():
                loss_history[name].append(val.item())

    pred_params_list = np.concatenate(pred_params_list, axis=0)
    pred_targets_list = np.concatenate(pred_targets_list, axis=0)

    plotter.plot_cov_matrix(pred_targets_list[:, :cfg_data['z_dim']])
    plotter.plot_hist(pred_targets_list[:, :cfg_data['z_dim']])
    plotter.plot_posterior(pos)

    np.save(join('test', f'epoch-{epoch}-params.npy'), pred_params_list)
    np.save(join('test', f'epoch-{epoch}-targets.npy'), pred_targets_list)
    loss_history = {name: np.mean(val) for name, val in loss_history.items()}

    return loss_history
