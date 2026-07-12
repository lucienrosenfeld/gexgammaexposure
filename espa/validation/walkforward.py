"""Purged, embargoed, expanding-window walk-forward (Section 22).

Quarterly refits on an expanding window. Purging removes training
observations whose information overlaps validation observations; with a
one-day holding window the overlap is the day itself, but the rolling
normalisations (252-day z-scores, 60-day EWMA losses) leak across the
boundary, which is what the 5-day embargo on *both sides* of each
validation fold kills.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from espa.config import DEFAULT_CONSTANTS, SpecConstants


@dataclass(frozen=True)
class WalkForwardFold:
    fold: int
    train_index: pd.Index
    validation_index: pd.Index


def walk_forward_folds(
    index: pd.Index,
    min_train: int = 252,
    constants: SpecConstants = DEFAULT_CONSTANTS,
) -> list[WalkForwardFold]:
    """Expanding-window folds over a chronologically sorted index.

    Each fold's validation block is one refit period
    (``constants.refit_frequency_days`` observations); training is
    everything before it minus the embargo. Because windows expand and
    validation blocks never precede training, the forward-side embargo
    is vacuous here, but the helper embargoes both sides so the same
    function stays correct if a diagnostic ever uses non-chronological
    blocks.
    """
    step = constants.refit_frequency_days
    emb = constants.embargo_days
    n = len(index)
    folds = []
    fold_no = 0
    start = min_train
    while start < n:
        val_idx = index[start : min(start + step, n)]
        train_end = start - emb  # purge + embargo before the validation block
        if train_end <= 0:
            start += step
            continue
        train_idx = index[:train_end]
        # embargo after the validation block (no-op for expanding windows)
        train_idx = train_idx.difference(
            index[min(start + step, n) : min(start + step + emb, n)]
        )
        folds.append(WalkForwardFold(fold_no, train_idx, val_idx))
        fold_no += 1
        start += step
    return folds
