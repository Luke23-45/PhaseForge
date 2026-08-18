# PhaseForge Fairness & Compute Accounting

| Method | Deployed Params | Excluded Heads* | Stage-2 Trainable | Active Params / Sample | Active Ratio vs BC | Forward FLOPs | Epochs (S1/S2) | Shared Stage-1 Source |
|---|---:|---:|---:|---:|---:|---:|:---:|:---:|
| `bc` | 206,983 | - | 206,983 | 206,983 | 1.000× | 407,687 | 100 / 0 | `self` |
| `bc_large` | 385,855 | - | 385,855 | 385,855 | 1.864× | 764,967 | 100 / 0 | `self` |
| `bc_robot_only` | 208,519 | - | 208,519 | 208,519 | 1.007× | 409,735 | 100 / 0 | `self` |
| `scratch_moe` | 382,646 | - | 382,646 | 243,354 | 1.176× | 478,626 | 0 / 100 | `none` |
| `warmstart_moe` | 382,646 | 34,823 | 210,486 | 243,354 | 1.176× | 478,626 | 0 / 100 | `bc` |
| `phase_pretrain_random_router` | 382,646 | 34,823 | 210,486 | 243,354 | 1.176× | 478,626 | 0 / 100 | `phaseforge` |
| `plain_encoder_phase_bootstrap` | 382,646 | 34,823 | 210,486 | 243,354 | 1.176× | 478,626 | 0 / 100 | `bc` |
| `phaseforge` | 382,646 | 35,597 | 210,486 | 243,354 | 1.176× | 478,626 | 100 / 100 | `self` |
| `pf_spherical_kmeans` | 382,646 | 35,597 | 210,486 | 243,354 | 1.176× | 478,626 | 0 / 100 | `phaseforge` |
| `pf_kmeans` | 382,646 | 35,597 | 210,486 | 243,354 | 1.176× | 478,626 | 0 / 100 | `phaseforge` |
| `pf_phase_head` | 382,646 | 35,597 | 210,486 | 243,354 | 1.176× | 478,626 | 0 / 100 | `phaseforge` |
| `pf_spherical` | 382,646 | 35,597 | 210,486 | 243,354 | 1.176× | 478,626 | 0 / 100 | `phaseforge` |
| `pf_random_random` | 382,646 | 35,597 | 210,486 | 243,354 | 1.176× | 478,626 | 0 / 100 | `phaseforge` |
| `pf_centroid_random` | 382,646 | 35,597 | 210,486 | 243,354 | 1.176× | 478,626 | 0 / 100 | `phaseforge` |
| `pf_ft` | 382,646 | 35,597 | 382,646 | 243,354 | 1.176× | 478,626 | 0 / 100 | `phaseforge` |
| `teacher_forced` | 381,872 | 35,597 | 209,712 | 207,757 | 1.004× | 409,236 | 0 / 100 | `phaseforge` |

\* *Note: Excluded Heads refers to detached Stage 1 ActionHead/PhaseHead parameters that are frozen and not in the Stage 2 computation graph or deployed policy.*
\* *Stage 1 pretraining runs are counted once per provider method (`phaseforge` or `bc`) and reused across consumers via dependency injection.*
