"""Phase-pretrained MoE with a randomly initialized router (C1 cell).

Completes the 2x2 factorial (issues register C1) isolating encoder-init vs
router-init effects:

    encoder x router     centroid (bootstrap)   random init
    phase-supervised     phaseforge             phase_pretrain_random_router
    plain (BC)           plain_encoder_phase_bootstrap   warmstart_moe

This cell shares the phase-supervised Stage 1 checkpoint with ``phaseforge``
(``resolve_checkpoint_source`` maps it there) and bootstraps the MoE with a
*random* router — structurally identical to :class:`WarmStartMoEModel`,
differing only in which Stage 1 checkpoint initializes the encoder.

Interpretation:
    vs ``phaseforge``        -> effect of the centroid router init.
    vs ``warmstart_moe``     -> effect of phase supervision in the
                                pretraining encoder (with a random router).
"""

from __future__ import annotations

from phaseforge.models.baselines.warmstart_moe import WarmStartMoEModel


class PhasePretrainRandomRouterModel(WarmStartMoEModel):
    """MoE whose encoder comes from phase-supervised pretraining, router random.

    Inherits the full warm-start behavior (Stage 1 encoder + action_head,
    experts initialized from the action head, router left randomly
    initialized, ``bootstrap_moe`` transition to Stage 2). The only
    difference from :class:`WarmStartMoEModel` is the Stage 1 checkpoint
    source, resolved by ``resolve_checkpoint_source``: this cell loads the
    phase-supervised checkpoint of ``phaseforge`` instead of the plain BC
    checkpoint.
    """
