from PIL import Image

from repalette.datasets import PreTrainDataset
from repalette.utils.transforms import sort_palette as sort_palette_by_hue


class GANDataset(PreTrainDataset):
    def _getitem(self, index):
        """
        @return: source_pair and target_pair - for generator; original_pair - for discriminator
        """
        (
            source_pair,
            target_pair,
        ) = super()._getitem(index)

        randomizer = self.get_randomizer()

        random_idx = randomizer.randint(0, len(self.query) - 1)
        rgb_image = self.query[random_idx]

        original_image = Image.open(rgb_image.path)
        original_palette = rgb_image.palette

        if self.sort_palette:
            original_palette = sort_palette_by_hue(original_palette)

        original_palette = Image.fromarray(original_palette)

        [original_image_aug] = self.image_transform(original_image, 0)
        [original_palette_aug] = self.palette_transform(original_palette, 0)

        original_pair = (
            original_image_aug,
            original_palette_aug,
        )

        return (
            source_pair,
            target_pair,
            original_pair,
        )
