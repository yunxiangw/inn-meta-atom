import torch
import numpy as np
from networks import build_inn_model


class Evaluation:
    def __init__(self, checkpoint_path, cfg):
        self.cfg = cfg
        self.model = self.load_model(checkpoint_path)

    def load_model(self, checkpoint_path):
        model = build_inn_model(**self.cfg['model'])
        checkpoint_dicts = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint_dicts['net'])
        model.to(self.cfg['device'])
        model.eval()
        return model

    def build_forward_inputs(self, x):
        """ Build input for forward prediction. """
        pad_dim = self.cfg['data']['input_dim'] - self.cfg['data']['x_dim']
        x = torch.Tensor(x)
        x = torch.cat((x, torch.zeros(x.shape[0], pad_dim)), dim=1) if pad_dim else x
        return x.to(self.cfg['device'])

    def build_backward_inputs(self, y):
        """ Build input for backward retrival. """
        pad_dim = self.cfg['data']['input_dim'] - self.cfg['data']['y_dim'] - self.cfg['data']['z_dim']
        z_dim = self.cfg['data']['z_dim']

        y = torch.Tensor(y)
        y = torch.cat((torch.zeros(y.shape[0], pad_dim), y), dim=1) if pad_dim else y
        y = torch.cat((torch.randn(y.shape[0], z_dim), y), dim=1)
        return y.to(self.cfg['device'])

    def forward_prediction(self, params):
        """ Model prediction for given parameters. """
        params = self.build_forward_inputs(params)

        with torch.no_grad():
            pred_targets, _ = self.model(params)
            pred_targets = pred_targets[:, -self.cfg['data']['y_dim']:].detach().cpu().numpy()

        return pred_targets

    def sample_posterior(self, target, n=128):
        """ Model prediction for a given target """
        target = np.tile(target, (n, 1))
        target = self.build_backward_inputs(target)

        with torch.no_grad():
            pred_params, _ = self.model(target, rev=True)
            pred_params = pred_params[:, :self.cfg['data']['x_dim']].detach().cpu().numpy()
            # Filtered out outliers
            pred_params = pred_params * 72.1688 + 175
            mask = np.all((pred_params >= 50) & (pred_params <= 300), axis=1)
            pred_params = pred_params[mask, :]
            pred_params = (pred_params - 175) / 72.1688  # Map back

        return pred_params