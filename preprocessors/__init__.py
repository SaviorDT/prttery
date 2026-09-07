"""Registry mapping ``--preprocessor`` CLI values to preprocessor classes."""

from __future__ import annotations

from torch.utils.data import Dataset

from preprocessors.base import PreprocessorBase
from preprocessors.copy_paste import CopyPastePreprocessor

PREPROCESSOR_REGISTRY: dict[str, type[PreprocessorBase]] = {
    "copy_paste": CopyPastePreprocessor,
}


def get_preprocessor_class(name: str) -> type[PreprocessorBase]:
    try:
        return PREPROCESSOR_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown preprocessor '{name}'. Choices: {sorted(PREPROCESSOR_REGISTRY)}") from None


def apply_preprocessors(
    train_dataset: Dataset,
    preprocessor_classes: tuple[type[PreprocessorBase], ...],
    *,
    copy_paste_count: int | None = None,
    copy_paste_seed: int | None = None,
) -> Dataset:
    """Apply configured preprocessor classes in order."""
    dataset = train_dataset
    for preprocessor_class in preprocessor_classes:
        if preprocessor_class is CopyPastePreprocessor:
            preprocessor = preprocessor_class(count=copy_paste_count, seed=copy_paste_seed)
        else:
            preprocessor = preprocessor_class()
        dataset = preprocessor.apply(dataset)
    return dataset
