import torch.nn as nn


activation_shortcuts = nn.ModuleDict(
    {
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "elu": nn.ELU(),
        "selu": nn.SELU(),
        "none": nn.Identity(),
    }
)
