import torch

from repalette.constants import L_RANGE, A_RANGE, B_RANGE


class Scaler:
    def __init__(self, old_range: torch.Tensor = None, new_range: torch.Tensor = None):
        self.old_range = old_range
        self.new_range = new_range
        if old_range is None:
            self.old_range = torch.as_tensor([L_RANGE, A_RANGE, B_RANGE], dtype=torch.float)
        if new_range is None:
            self.new_range = torch.as_tensor([[-1, 1]] * 3, dtype=torch.float)

    def transform(self, img: torch.Tensor):
        """
        Scales image into bounds, determined by `new_range` parameter.

        Parameters
        ----------
        img : torch.Tensor
            Image to be transformed.

        Returns
        -------
        torch.Tensor
            Transformed image.
        """
        return normalize_img(img, self.old_range, self.new_range)

    def inverse_transform(self, img: torch.Tensor):
        return normalize_img(img, self.new_range, self.old_range)


def normalize_img(img: torch.Tensor, old_range: torch.Tensor, new_range: torch.Tensor):
    new_img = img - torch.mean(old_range, dim=-1)[:, None, None]
    coef = (new_range[:, 1] - new_range[:, 0]) / (old_range[:, 1] - old_range[:, 0])
    new_img *= coef[:, None, None]
    new_img += torch.mean(new_range, dim=-1)[:, None, None]
    return new_img
