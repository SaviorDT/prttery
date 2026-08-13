"""Shared pretrained-encoder freeze/unfreeze mechanics for ``--freeze-encoder``.

``train.normal`` and ``train.multiclass`` are deliberately parallel, not
shared, training loops (see their docstrings) -- but the freeze/unfreeze
mechanics below are small, purely mechanical, and must stay byte-for-byte
identical between them (a drift here would silently change what "frozen"
means for one model family and not the other), so they live here once and
both loops import them instead of each keeping its own copy.

Only the *mechanics* (mutating the model/optimizer) live here. Deciding
*when* to call these -- the per-epoch no-improvement counters -- stays
inline in each training loop, next to that loop's own early-stopping/
lr-decay bookkeeping it has to interleave with.
"""

from __future__ import annotations

import itertools

from models.base import ModelBase


def freeze_encoder(model: ModelBase) -> None:
    """Freeze ``model``'s pretrained encoder: disable gradients and pin its
    BatchNorm/LayerNorm running stats (via eval mode), so the pretrained
    ImageNet features stay untouched while only the decoder trains.

    ``ModelBase.train()`` re-applies eval() to these modules after every
    step (it would otherwise be undone by the ``self.net.train()`` call at
    the top of that method), so the freeze holds until unfreeze_encoder()
    is called.

    ``ModelBase.create()`` builds the optimizer from *every* net parameter
    (``Adam(self.net.parameters(), lr=lr)``), before this function ever
    runs -- so the encoder's parameters are already sitting in
    ``optimizer.param_groups[0]`` (requires_grad=False just makes .step()
    skip them; PyTorch never garbage-collects a param out of a group on its
    own). They're spliced back out of that group here so unfreeze_encoder()
    can later hand them to add_param_group() without PyTorch rejecting them
    as "already in a param group".
    """
    encoder_modules = model.encoder_modules()
    encoder_param_ids = {id(p) for module in encoder_modules for p in module.parameters()}
    for module in encoder_modules:
        module.requires_grad_(False)
        module.eval()
    decoder_group = model.optimizer.param_groups[0]
    decoder_group["params"] = [p for p in decoder_group["params"] if id(p) not in encoder_param_ids]
    model._frozen_encoder_modules = encoder_modules


def unfreeze_encoder(model: ModelBase, encoder_lr_factor: float) -> None:
    """Unfreeze ``model``'s encoder: re-enable gradients and add it to the
    optimizer as a new, discriminative-LR param group.

    The optimizer is not rebuilt -- decoder parameters keep whatever Adam
    momentum/variance state they've already accumulated -- the encoder's
    parameters are simply added as a new group via
    ``optimizer.add_param_group()``, at ``encoder_lr_factor`` times the
    decoder's *current* lr (param_groups[0], which may already have been
    decayed by --lr-mode decreasing), not the run's original --lr. Using the
    current lr keeps the encoder's rate a fixed fraction of the decoder's
    even after further --lr-mode decreasing decays (that logic scales every
    param group uniformly), instead of drifting out of proportion.
    """
    encoder_modules = model.encoder_modules()
    for module in encoder_modules:
        module.requires_grad_(True)
    model._frozen_encoder_modules = []

    decoder_lr = model.optimizer.param_groups[0]["lr"]
    encoder_params = list(itertools.chain.from_iterable(module.parameters() for module in encoder_modules))
    model.optimizer.add_param_group({"params": encoder_params, "lr": decoder_lr * encoder_lr_factor})
