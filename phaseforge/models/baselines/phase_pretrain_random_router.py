"""Phase-pretrained MoE with a randomly initialized router (C1 cell).

Completes the 2x2 factorial (issues register C1) isolating encoder-init vs
router-init effects, with expert initialization held fixed at the canonical
R50 50% partial warm-start:

    encoder x router     centroid (bootstrap)   random init
    phase-supervised     phaseforge             phase_pretrain_random_router
    plain (BC)           plain_encoder_phase_bootstrap   warmstart_moe*

    (* behavioral baseline: warmstart_moe keeps the standard full warm start;
     the R50-matched factorial is {encoder source} x {router init} at fixed
     partial-warm expert init.)

This cell shares the phase-supervised Stage 1 checkpoint with the canonical
``phaseforge`` (``resolve_checkpoint_source`` maps it there) and bootstraps
the MoE with a *random* router. Its model config sets
``expert_init.type=partial_warm`` (drop_rate 0.5, seed tied to the training
seed) so it is an exact R50 factorial control for the H1 router-init claim.

Interpretation (vs the canonical partial-warm phaseforge):
    vs ``phaseforge``        -> effect of the centroid router init.
    vs ``plain_encoder_phase_bootstrap`` -> effect of phase supervision in
                                the pretraining encoder (same partial-warm
                                experts, same centroid router).
"""

from __future__ import annotations

from phaseforge.models.baselines.warmstart_moe import WarmStartMoEModel


class PhasePretrainRandomRouterModel(WarmStartMoEModel):
    """MoE whose encoder comes from phase-supervised pretraining, router random.

    Inherits the config-driven bootstrap from :class:`WarmStartMoEModel`
    (Stage 1 encoder + action_head, experts initialized per ``expert_init``,
    router left randomly initialized, ``bootstrap_moe`` transition to
    Stage 2). Two differences from a default :class:`WarmStartMoEModel`:

    1. The Stage 1 checkpoint source — resolved by
       ``resolve_checkpoint_source`` to the phase-supervised checkpoint of
       ``phaseforge`` instead of the plain BC checkpoint.
    2. The registered model config pins ``expert_init`` to the canonical
       50% partial warm-start, matching the proposed method.
    """
