from torch.utils.data import DataLoader


class ShuffleDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(ShuffleDataLoader, self).__init__(*args, **kwargs)

    def shuffle(self, set_shuffle=True):
        self.dataset.shuffle(set_shuffle)
        return self
