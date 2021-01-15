from torch.utils.data import Dataset
import abc
from random import Random


class AbstractDataset(abc.ABC, Dataset):
    """
    Base class for all repalette dataset
    """

    def __init__(self, random_seed, **kwargs):
        self.random_seed = random_seed
        self.consistent_random_seed = Random().random()

        self._kwargs = kwargs

    def __getitem__(self, index):
        """
        :param index: index of item to get from the dataset
        :return: image of shape [3, self.resize] and palette of shape [3, 1, 6]
        """
        return self._getitem(index)

    @abc.abstractmethod
    def _getitem(self, index):
        raise NotImplementedError

    def __len__(self):
        """
        :return: dataset length
        """
        return self._len()

    @abc.abstractmethod
    def _len(self):
        raise NotImplementedError

    def split(self, test_size=0.2, shuffle=True, random_seed=None):
        randomizer = self.get_randomizer(random_seed=random_seed)
        return self._split(test_size=test_size, shuffle=shuffle, randomizer=randomizer)

    @abc.abstractmethod
    def _split(self, test_size, shuffle, randomizer: Random):
        raise NotImplementedError

    def shuffle(self, to_shuffle=True, random_seed=None):
        """
        Shuffles data.
        :param to_shuffle: if to shuffle
        :param random_seed: random seed for shuffling. Use "lock" to get consistent results.
        """
        randomizer = self.get_randomizer(random_seed=random_seed)
        self._shuffle(to_shuffle=to_shuffle, randomizer=randomizer)

    @abc.abstractmethod
    def _shuffle(self, to_shuffle, randomizer: Random):
        raise NotImplementedError

    def get_randomizer(self, random_seed=None):
        if random_seed is None:
            random_seed = self.random_seed
        elif random_seed == "lock":
            random_seed = self.consistent_random_seed

        random = Random(random_seed)

        return random


class AbstractQueryDataset(AbstractDataset):
    @abc.abstractmethod
    def _getitem(self, index):
        raise NotImplementedError

    def __init__(self, query, random_seed=None, **kwargs):
        super().__init__(random_seed=random_seed, **kwargs)

        self.query = query
        self.correct_order_query = self.query

    def _len(self):
        return len(self.query)

    def _split(self, test_size, shuffle, randomizer: Random):
        train_query, test_query = self._query_split(
            test_size=test_size, shuffle=shuffle, randomizer=randomizer
        )

        train = self.__class__(query=train_query, random_seed=self.random_seed, **self._kwargs)
        test = self.__class__(query=test_query, random_seed=self.random_seed, **self._kwargs)

        return train, test

    def _query_split(self, test_size, shuffle, randomizer):
        query = self.query

        if shuffle:
            randomizer.shuffle(query)

        train_query = query[: int(len(query) * (1 - test_size))]
        test_query = query[int(len(query) * (1 - test_size)) :]

        return train_query, test_query

    def _shuffle(self, to_shuffle, randomizer: Random):
        if to_shuffle:
            randomizer.shuffle(self.query)
        else:
            self.query = self.correct_order_query


class AbstractRecolorDataset(AbstractQueryDataset):
    def __init__(
        self,
        query,
        multiplier,
        random_seed=None,
        train_kwargs: dict = None,
        test_kwargs: dict = None,
        **kwargs,
    ):
        super().__init__(query=query, random_seed=random_seed, **kwargs)

        self._kwargs = kwargs

        self.multiplier = multiplier
        self.hue_pairs = self._make_hue_pairs(multiplier)
        self.n_pairs = len(self.hue_pairs)
        self.correct_order_hue_pairs = self.hue_pairs

        self.train_kwargs = train_kwargs
        self.test_kwargs = test_kwargs

    def _len(self):
        return len(self.query) * self.n_pairs

    @abc.abstractmethod
    def _make_hue_pairs(self, multiplier):
        raise NotImplementedError

    def _split(self, test_size, shuffle, randomizer: Random):
        train_query, test_query = self._query_split(
            test_size=test_size, shuffle=shuffle, randomizer=randomizer
        )

        train_kwargs = self._kwargs
        train_kwargs.update(self.train_kwargs)

        train = self.__class__(
            query=train_query,
            multiplier=self.multiplier,
            random_seed=self.random_seed,
            **train_kwargs,
        )

        test_kwargs = self._kwargs
        test_kwargs.update(self.test_kwargs)

        test = self.__class__(
            query=test_query,
            multiplier=self.multiplier,
            random_seed=self.random_seed,
            **test_kwargs,
        )

        return train, test

    def _shuffle(self, to_shuffle, randomizer: Random):
        if to_shuffle:
            randomizer.shuffle(self.query)
            randomizer.shuffle(self.hue_pairs)
        else:
            self.query = self.correct_order_query
            self.hue_pairs = self.correct_order_hue_pairs

        return self
