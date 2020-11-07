import torchvision.transforms.functional as TF
from skimage.color import rgb2lab, lab2rgb
from PIL import Image
import numpy as np


def smart_hue_adjust(img, hue_shift: float, lab=True) -> np.float32:
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
        if img.dtype in [np.float16, np.float32, np.float64]:
            pil_img = Image.fromarray(img)
            np_img = img
        elif img.dtype in [np.int16, np.int32, np.int64]:
            pil_img = Image.fromarray(img)
            np_img = img.astype(np.float) / 255.0
        else:
            raise ValueError(
                "Numpy array dtype must be one of:\n"
                "np.float16, np.float32, np.float64, np.int16, np.int32, np.int64"
            )
    else:
        pil_img = img
        np_img = np.array(img).astype(np.float) / 255.0

    # get original luminance
    img_LAB = rgb2lab(np_img)
    luminance = img_LAB[:, :, 0]

    # shift hue
    img_shifted = np.array(TF.adjust_hue(pil_img, hue_shift)).astype(np.float) / 255.0
    img_shifted_LAB = rgb2lab(img_shifted)
    img_shifted_LAB[:, :, 0] = luminance  # restore original luminance
    if lab:
        return img_shifted_LAB
    img_augmented = (lab2rgb(img_shifted_LAB) * 255.0).astype(int)

    return img_augmented
