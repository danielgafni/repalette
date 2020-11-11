from torch.utils.data import DataLoader


class ShuffleDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(ShuffleDataLoader, self).__init__(*args, **kwargs)

    def shuffle(self, to_shuffle=True):
        self.dataset.shuffle(to_shuffle)
        return self
