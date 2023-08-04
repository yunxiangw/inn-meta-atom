import torch
import torch.nn as nn

from FrEIA.modules import *
from FrEIA.framework import *
from torchvision.models import resnet34


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.leaky_relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = self.linear1(x)
        out = self.leaky_relu(out)
        out = self.linear2(out)

        out += 0.1 * x
        out = self.leaky_relu(out)

        return out


class Subnet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear_in = nn.Linear(input_dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, output_dim)
        self.layer1 = ResBlock(hidden_dim)
        self.layer2 = ResBlock(hidden_dim)
        self.leaky_relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = self.linear_in(x)
        out = self.leaky_relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.linear_out(out)

        return out


def build_inn_model(
        input_dim,
        n_blocks,
        exponent_clamping,
        hidden_layers,
        verbose
):

    def subnet_func(c_in, c_out):
        model = []
        prev = c_in
        layers = hidden_layers + [c_out]
        for curr in layers:
            model.append(nn.Linear(prev, curr))
            model.append(nn.LeakyReLU())
            prev = curr
        model.pop()
        return nn.Sequential(*model)

    """
    def subnet_func(input_dim, output_dim):
        return Subnet(input_dim, 64, output_dim)
    """

    nodes = [InputNode(input_dim, name='input')]
    for i in range(n_blocks):
        nodes.append(Node(nodes[-1],
                          GLOWCouplingBlock,
                          {'subnet_constructor': subnet_func,
                           'clamp': exponent_clamping},
                          name=f'coupling_{i}'))
        nodes.append(Node(nodes[-1],
                          PermuteRandom,
                          {'seed': i},
                          name=f'permute_{i}'))
    nodes.append(OutputNode(nodes[-1], name='output'))

    model = ReversibleGraphNet(nodes, verbose=verbose)

    return model
