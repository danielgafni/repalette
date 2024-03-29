import torch


def pied(x: torch.Tensor, y: torch.Tensor):
    """
    Computes permutation invariant euclidean distance between two sets of vectors.
    :param x: first set of vectors
    :type x: torch.Tensor of shape [batch_size, n_vectors, vector_dim_size] (batch_size optional)
    :param y: second set of vectors
    :type y: torch.Tensor of shape [batch_size, n_vectors, vector_dim_size] (batch_size optional)
    :return: permutation invariant euclidean distance
    :rtype: torch.FloatTensor
    """
    x = x.unsqueeze(-2).float()
    y = y.unsqueeze(-3).float()

    return torch.sqrt(((x - y) ** 2).sum())
