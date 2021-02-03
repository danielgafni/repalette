import numpy as np
import torch
from PIL import Image
from skimage.color import hsv2rgb, lab2rgb, rgb2hsv, rgb2lab
from torchvision.transforms import functional as TF

from repalette.constants import A_RANGE, B_RANGE, L_RANGE


class LABNormalizer:
    def __init__(
        self,
        old_range: torch.Tensor = None,
        new_range: torch.Tensor = None,
    ):
        if old_range is None:
            self.old_range = torch.as_tensor(
                [L_RANGE, A_RANGE, B_RANGE],
                dtype=torch.float,
            )
        else:
            self.old_range = old_range
        if new_range is None:
            self.new_range = torch.as_tensor([[-1, 1]] * 3, dtype=torch.float)
        else:
            self.new_range = new_range

    def to(
        self,
        device: torch.device = torch.device("cpu"),
    ):
        self.old_range = self.old_range.to(device)
        self.new_range = self.new_range.to(device)

    def normalize(self, img: torch.Tensor):
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


def normalize_img(
    img: torch.Tensor,
    old_range: torch.Tensor,
    new_range: torch.Tensor,
):
    new_img = img - torch.mean(old_range, dim=-1)[:, None, None]
    coef = (new_range[:, 1] - new_range[:, 0]) / (old_range[:, 1] - old_range[:, 0])
    new_img *= coef[:, None, None]
    new_img += torch.mean(new_range, dim=-1)[:, None, None]
    return new_img


class FullTransform:
    """Wrapping class for user `torchvision.transforms` transformation followed by hue adjustment
    and casting to torch tensor. Returns a list of images, same size as `hue_shifts`.
    """

    def __init__(self, transform=None, normalize=True):
        self.transform = transform
        self.normalizer = LABNormalizer() if normalize else None

    def __call__(self, img, *hue_shifts):
        """
        Shifts hue for an image, preserving its luminance
        :param img: Input image
        :type img: a `PIL` image
        :param hue_shifts: hue shifts in [-0.5, 0.5] interval
        :type hue_shifts: float
        :return: hue-shifted images with original luminance
        :rtype: list[np.float32]
        """
        if self.transform is not None:
            img = self.transform(img)
        augmented_imgs = []
        for hue_shift in hue_shifts:
            aug_img = smart_hue_adjust(img, hue_shift)
            aug_img = TF.to_tensor(aug_img).to(torch.float)
            if self.normalizer:
                aug_img = self.normalizer.normalize(aug_img)
            augmented_imgs.append(aug_img)
        return augmented_imgs


def smart_hue_adjust(img, hue_shift: float, lab=True):
    """
    Shifts hue for an image, preserving its luminance
    :param img: Input image
    :type img: a `PIL` image or `numpy.ndarray`
    :param hue_shift: hue shift in [-0.5, 0.5] interval
    :type hue_shift: float
    :param lab: if True, returns image in LAB format
    :return: hue-shifted image with original luminance
    :rtype: np.float32
    """

    # if submitted numpy array - convert to PIL.Image
    if type(img) == np.ndarray:
        if img.dtype in [
            np.float16,
            np.float32,
            np.float64,
        ]:
            pil_img = Image.fromarray(img)
            np_img = img
        elif img.dtype in [
            np.int16,
            np.int32,
            np.int64,
        ]:
            pil_img = Image.fromarray(img)
            np_img = img.astype("float") / 255.0
        else:
            raise ValueError(
                "Numpy array dtype must be one of:\n" "np.float16, np.float32, np.float64, np.int16, np.int32, np.int64"
            )
    else:
        pil_img = img
        np_img = np.array(img).astype("float") / 255.0

    # get original luminance
    img_LAB = rgb2lab(np_img)
    luminance = img_LAB[:, :, 0]

    # shift hue
    img_shifted = np.array(TF.adjust_hue(pil_img, hue_shift)).astype("float") / 255.0
    img_shifted_LAB = rgb2lab(img_shifted)
    img_shifted_LAB[:, :, 0] = luminance  # restore original luminance
    if lab:
        return img_shifted_LAB.astype(int)
    img_augmented = (lab2rgb(img_shifted_LAB) * 255.0).astype(int)

    return img_augmented


def sort_palette(palette):
    """
    Sorts palette by hue
    Parameters
    ----------
    palette
        numpy array of shape [1, :, 3]. Must be an RGB image.
    Returns
        sorted palette of shape [1. :, 3] and dtype np.uint8
    -------

    """
    palette_hsv = rgb2hsv(palette)
    sort_args = np.argsort(palette_hsv[..., 0], axis=1).flatten()
    palette_sorted = (hsv2rgb(palette_hsv[:, sort_args, :]) * 255).astype("uint8")
    return palette_sorted
