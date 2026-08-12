"""Abstract base class for train-set preprocessors (data augmentation steps
applied to an already-built train ``Dataset``, selected via ``--preprocessor``).

A concrete preprocessor only has to implement ``apply``, which takes the
train dataset and returns a (possibly different) dataset to use in its
place -- e.g. one that adds synthetic samples on top of the original ones.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch.utils.data import Dataset


class PreprocessorBase(ABC):
    """Common interface shared by every preprocessor."""

    @abstractmethod
    def apply(self, train_dataset: Dataset) -> Dataset:
        """Return the dataset to train on, derived from ``train_dataset``."""
        raise NotImplementedError
