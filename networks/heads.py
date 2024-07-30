import torch
from torch import nn
from torch.nn import Conv2d


class BatchNorm1dNoBias(nn.BatchNorm1d):
    """
    https://github.com/AndrewAtanov/simclr-pytorch
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias.requires_grad = False

class MLPHead(nn.Module):
    def __init__(self, args, in_channels, mlp_hidden_size, projection_size, layer_size = 1):
        super(MLPHead, self).__init__()
        if layer_size == 0:
            self.net = nn.Linear(in_channels, projection_size)
        elif layer_size == 1:
            self.net = nn.Sequential(
                nn.Linear(in_channels, mlp_hidden_size),
                nn.BatchNorm1d(mlp_hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(mlp_hidden_size, projection_size)
            )
        elif layer_size == 2:
            self.net = nn.Sequential(
                nn.Linear(in_channels, mlp_hidden_size),
                nn.BatchNorm1d(mlp_hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(mlp_hidden_size, mlp_hidden_size),
                nn.BatchNorm1d(mlp_hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(mlp_hidden_size, projection_size)
            )
        elif layer_size == 3:
            self.net = nn.Sequential(
                nn.Linear(in_channels, mlp_hidden_size),
                nn.BatchNorm1d(mlp_hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(mlp_hidden_size, mlp_hidden_size),
                nn.BatchNorm1d(mlp_hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(mlp_hidden_size, mlp_hidden_size),
                nn.BatchNorm1d(mlp_hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(mlp_hidden_size, projection_size)
        )
    def forward(self, x):
        return self.net(x)
