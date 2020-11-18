import torch
from torchvision.utils import make_grid
from skimage.color import lab2rgb
import warnings


def lab_batch_to_rgb_image_grid(lab_batch, padding=2, pad_value=0):
    """
    Makes image grid ready to be logged to TensorBoard from a LAB images batch.
    :param lab_batch: images batch in LAB of shape [batch_size, 3, :, :].
    :param padding: padding thickness between images in pixels
    :param pad_value: padding value
    :return: torch.Tensor, images stacked with `torchvision.utils.make_grid`.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid = make_grid(
            torch.stack(
                [
                    torch.from_numpy(lab2rgb(lab_image.cpu()))
                    for lab_image in lab_batch.permute(0, 2, 3, 1)
                ]
            ).permute(0, 3, 1, 2),
            padding=padding,
            pad_value=pad_value,
        )
    return grid
