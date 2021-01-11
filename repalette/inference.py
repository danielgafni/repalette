from PIL import Image
from skimage.color import lab2rgb, rgb2lab
import torch
from torchvision.transforms.functional import to_tensor
from torchvision import transforms

from repalette.utils.transforms import LABNormalizer


def recolor_image(image, palette, generator):
    generator.eval()

    normalizer = LABNormalizer()
    image_size = image.size[1], image.size[0]

    image_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
        ]
    )

    palette_transform = transforms.Compose([])

    image_transformed = image_transform(image)
    palette_transformed = palette_transform(palette)

    image_lab = rgb2lab(image_transformed)
    palette_lab = rgb2lab(palette_transformed)

    image_tensor = to_tensor(image_lab).float()
    palette_tensor = to_tensor(palette_lab).float()

    image_normalized = normalizer.normalize(image_tensor).unsqueeze(0)
    palette_normalized = normalizer.normalize(palette_tensor).view(1, -1)

    recolored_image_ab = generator(image_normalized, palette_normalized).detach().cpu()

    original_luminance = image_normalized.clone()[:, 0:1, ...]
    recolored_img_lab_normalized = torch.cat(
        (
            original_luminance,
            recolored_image_ab,
        ),
        dim=1,
    )
    recolored_image_lab = normalizer.inverse_transform(recolored_img_lab_normalized)
    recolored_image = lab2rgb(recolored_image_lab.squeeze(0).permute(1, 2, 0))

    recolored_image = Image.fromarray(
        (recolored_image * 255).astype("uint8").reshape(image_size[0], image_size[1], 3)
    )

    return recolored_image
