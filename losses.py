import torch
import torch.nn as nn


class MMDLoss(nn.Module):
    def __init__(self, widths_exponents):
        super().__init__()
        self.widths_exponents = widths_exponents

    def mmd_multiscale(self, x, y):
        device = x.get_device()

        xx, yy, xy = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())

        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))

        dxx = torch.clamp(rx.t() + rx - 2. * xx, 0, float('inf'))
        dyy = torch.clamp(ry.t() + ry - 2. * yy, 0, float('inf'))
        dxy = torch.clamp(rx.t() + ry - 2. * xy, 0, float('inf'))

        XX, YY, XY = (
            torch.zeros(xx.shape).to(device),
            torch.zeros(xx.shape).to(device),
            torch.zeros(xx.shape).to(device)
        )

        """
        for C, a in self.widths_exponents:
            XX += C ** a * ((C + dxx) / a) ** -a
            YY += C ** a * ((C + dyy) / a) ** -a
            XY += C ** a * ((C + dxy) / a) ** -a
        """

        for a in self.widths_exponents:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1

        return torch.mean(XX + YY - 2 * XY)

    def forward(self, inputs, targets):
        return self.mmd_multiscale(inputs, targets)


class BidirectionalLoss(nn.Module):
    def __init__(
            self,
            input_dim,
            x_dim,
            y_dim,
            z_dim,
            weight,
            mmd_kernel
    ):
        super().__init__()
        self.input_dim = input_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim

        self.w = weight
        self.fit_loss = nn.MSELoss(reduction='sum')
        self.mmdf_loss = MMDLoss(mmd_kernel['forward'])
        self.mmdb_loss = MMDLoss(mmd_kernel['backward'])

    def forward_fit(self, pred_zy, zy):
        l = self.w['forward_fit'] * self.fit_loss(pred_zy[:, self.z_dim:], zy[:, self.z_dim:])
        return l / zy.size(dim=0)

    def forward_mmd(self, pred_zy, zy):
        # Shorten output, and remove gradients wrt y, for latent loss
        pred_block_y = torch.cat((pred_zy[:, :self.z_dim], pred_zy[:, -self.y_dim:].detach()), dim=1)
        zy = torch.cat((zy[:, :self.z_dim], zy[:, -self.y_dim:]), dim=1)

        return self.w['forward_mmd'] * self.mmdf_loss(pred_block_y, zy)
    
    def reconstruct(self, x_reconstruct, x):
        l = self.w['backward_fit'] * self.fit_loss(x_reconstruct, x)
        return l / x.size(dim=0)

    def backward_mmd(self, pred_x, x):
        return self.w['backward_mmd'] * self.mmdb_loss(pred_x, x)

    def forward(self, x, zy, pred_x, pred_zy, x_reconstruct, mode='init'):
        losses = {
            'forward_fit': self.forward_fit(pred_zy, zy),
            'forward_mmd': self.forward_mmd(pred_zy, zy),
            'reconstruct': self.reconstruct(x_reconstruct, x)
        }
        if mode == 'train':
            losses['backward_mmd'] = self.backward_mmd(pred_x, x)

        return losses
