import torch.nn as nn


activation_shortcuts = {
    'relu': nn.ReLU(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'elu': nn.ELU(),
    'selu': nn.SELU(),
    None: nn.Identity(),
}
