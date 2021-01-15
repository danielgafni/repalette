from torch.utils.data import DataLoader


class ShuffleDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(ShuffleDataLoader, self).__init__(*args, **kwargs)

    def shuffle(self, to_shuffle=True, random_seed=None):
        self.dataset.shuffle(to_shuffle, random_seed=random_seed)
        return self
