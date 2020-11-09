import torch
from torchvision.utils import make_grid
from skimage.color import lab2rgb


def lab_batch_to_rgb_image_grid(lab_batch, padding=2, pad_value=0):
    """
    Makes image grid ready to be logged to TensorBoard from a LAB images batch.
    :param lab_batch: images batch in LAB of shape [batch_size, 3, :, :].
    :param padding: padding thickness between images in pixels
    :param pad_value: padding value
    :return: torch.Tensor, images stacked with `torchvision.utils.make_grid`.
    """
    grid = make_grid(
        torch.stack(
            [
                torch.from_numpy(lab2rgb(lab_image.permute(1, 2, 0).cpu())).permute(
                    2, 0, 1
                )
                for lab_image in lab_batch
            ]
        ),
        padding=padding,
        pad_value=pad_value,
    )
    return grid
