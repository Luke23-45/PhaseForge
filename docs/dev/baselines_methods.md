I reviewed the report and traced the implementations/configurations. The baseline tests also pass: 59/59.

Report reviewed: [DEBUG_RUN_BASELINE_COMPARISON_REPORT.md](C:/Users/Hellx/Documents/Programming/python/Project/Neryva/PhaseForge/docs/DEBUG_RUN_BASELINE_COMPARISON_REPORT.md)

## Verified baseline methods

| Method | Implementation | Actual behavior |
|---|---|---|
| BC standard | [bc.py](C:/Users/Hellx/Documents/Programming/python/Project/Neryva/PhaseForge/phaseforge/models/baselines/bc.py) | Encoder → action head; no phase or routing |
| BC Large | Same `BehaviorCloningModel`; [bc_large.yaml](C:/Users/Hellx/Documents/Programming/python/Project/Neryva/PhaseForge/phaseforge/config/models/baselines/bc_large.yaml) | Wider dense encoder/action head; not a separate Python class |
| BC Robot Only | Same BC implementation with robot-only data | Data/schema negative control, not a separate model |
| Scratch MoE | [scratch_moe.py](C:/Users/Hellx/Documents/Programming/python/Project/Neryva/PhaseForge/phaseforge/models/baselines/scratch_moe.py) | Trainable encoder, random router, randomly initialized experts; Stage 2 only |
| Warmstart MoE | [warmstart_moe.py](C:/Users/Hellx/Documents/Programming/python/Project/Neryva/PhaseForge/phaseforge/models/baselines/warmstart_moe.py) | BC-pretrained encoder; random router; experts copied from the BC action head |
| Phase Pretrain Random Router | [phase_pretrain_random_router.py](C:/Users/Hellx/Documents/Programming/python/Project/Neryva/PhaseForge/phaseforge/models/baselines/phase_pretrain_random_router.py) | Phase-supervised encoder; random router; **R50-matched 50% partial warm-start experts** (config-driven `expert_init` inherited from `WarmStartMoEModel`) |
| Plain Encoder Phase Bootstrap | [plain_encoder_phase_bootstrap.py](C:/Users/Hellx/Documents/Programming/python/Project/Neryva/PhaseForge/phaseforge/models/baselines/plain_encoder_phase_bootstrap.py) | BC encoder; phase-centroid router initialization; **R50-matched 50% partial warm-start experts** (config-driven `expert_init`) |
| Teacher Forced | [teacher_forced.py](C:/Users/Hellx/Documents/Programming/python/Project/Neryva/PhaseForge/phaseforge/models/baselines/teacher_forced.py) | Ground-truth phase routing during training; predicted phase routing during evaluation |
| Oracle MoE | [oracle_moe.py](C:/Users/Hellx/Documents/Programming/python/Project/Neryva/PhaseForge/phaseforge/models/baselines/oracle_moe.py) | Ground-truth phase routing during training; rollout router is untrained, so it is not a deployable policy |

## Additional PhaseForge ablation cells

These use the shared `PhaseBootstrappedMoE` implementation in [phase_moe.py](C:/Users/Hellx/Documents/Programming/python/Project/Neryva/PhaseForge/phaseforge/models/phase_moe.py); all router-bearing cells except the random-experts pair are pinned to the canonical R50 50% partial warm-start:

- `pf_spherical_kmeans`, `pf_kmeans`, `pf_phase_head`, `pf_spherical`, `pf_ft` — partial warm (0.5)
- `pf_random_random`, `pf_centroid_random` — random experts by design (the "fully random experts" cells)

Their router/expert initialization is defined in the corresponding files under [config/models/baselines](C:/Users/Hellx/Documents/Programming/python/Project/Neryva/PhaseForge/phaseforge/config/models/baselines).

The ablation manifest also contains configuration-override cells rather than new implementations:

- `pf_k3`, `pf_k12`
- `pf_corrupt_25`, `pf_corrupt_50`, `pf_shuffle_control`
- `pf_one_warm_plus_random`
- `pf_full_warm`, `pf_drop00`, `pf_drop25`, `pf_drop75`, `pf_drop100`

Removed after the canonical R50 migration (2026-08-22): `pf_random_warm`,
`phaseforge_e6`, `warmstart_r50` (each recreated the canonical method), and
`pf_jitter_00`/`pf_jitter_10` (jitter is inert under partial warm; the
drop-rate sweep subsumes the endpoints).

## Important report corrections

1. The report says PhaseForge’s 74% is the highest single-seed result across all baselines. That is incorrect: `pf_centroid_random` has a reported 92% result on seed 43.

2. The report’s 13-method table is not a complete inventory of repository methods. It omits `pf_phase_head`, `pf_ft`, and several ablation cells; the final protocol’s active baseline inventory is defined by `experiments/five_task.json` and the separate Lift ablation manifest.

3. BC Large and BC Robot Only are not separate model classes. Both use the standard BC implementation; they differ through configuration/data.

4. Oracle MoE should remain clearly labeled as a training/routing diagnostic. The implementation explicitly requires phase labels during training and does not train the inference router.

5. The report’s PhaseForge-versus-BC comparison must remain directional unless the matched evaluation-bank verification mentioned in the report has actually been completed.

6. The `pf_centroid_random` versus PhaseForge result should remain a near-tie, not evidence that either method is superior, because the 0.92 seed is explicitly flagged as an outlier.

Overall, the report’s main conclusions about the matched PhaseForge/`pf_centroid_random` comparison, routing collapse in Teacher Forced/Oracle, and the factorial controls are consistent with the code. The method inventory and the “highest single-seed result” statement need correction before the final experiment.
