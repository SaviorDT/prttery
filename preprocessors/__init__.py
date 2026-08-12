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
    names: list[str],
    *,
    copy_paste_count: int | None = None,
    copy_paste_seed: int | None = None,
) -> Dataset:
    """Apply each preprocessor in ``names``, in order, to ``train_dataset``.

    Only ``copy_paste`` exists today, so its own args (``copy_paste_count``/
    ``copy_paste_seed``) are threaded through explicitly here rather than via
    a generic per-name config dict; once a second preprocessor needs its own
    args, this should grow a proper per-name config shape instead.
    """
    dataset = train_dataset
    for name in names:
        preprocessor_class = get_preprocessor_class(name)
        if name == "copy_paste":
            preprocessor = preprocessor_class(count=copy_paste_count, seed=copy_paste_seed)
        else:
            preprocessor = preprocessor_class()
        dataset = preprocessor.apply(dataset)
    return dataset
