
[v2-gates] g3: same-wave same-bank re-baseline (EXP-102 + EXP-101)
[v2-gates] g3: phaseforge-sweep --manifest /content/PhaseForge/experiments/lift_ablation.json --methods bc,phaseforge --seeds 42,43,44 --outputs outputs/v2_g3
[runner] commit gate: 446e947

[runner] plan (15 steps, outputs base: /content/PhaseForge/outputs/v2_g3)
    1. phaseforge seed=42 stage1                        pending
    2. phaseforge seed=42 stage2                        pending
    3. phaseforge seed=42 eval                          pending
    4. phaseforge seed=43 stage1                        pending
    5. phaseforge seed=43 stage2                        pending
    6. phaseforge seed=43 eval                          pending
    7. phaseforge seed=44 stage1                        pending
    8. phaseforge seed=44 stage2                        pending
    9. phaseforge seed=44 eval                          pending
   10. bc seed=42 stage1                                pending
   11. bc seed=42 eval                                  pending
   12. bc seed=43 stage1                                pending
   13. bc seed=43 eval                                  pending
   14. bc seed=44 stage1                                pending
   15. bc seed=44 eval                                  pending

[runner] $ /content/PhaseForge/.venv/bin/phaseforge-train project.log_level=WARNING models=phaseforge train=stage1 project.seed=42 project.output_dir=/content/PhaseForge/outputs/v2_g3 project.method=phaseforge train.early_stopping.enabled=false

[1/15] OK phaseforge seed=42 stage1

[runner] $ /content/PhaseForge/.venv/bin/phaseforge-train project.log_level=WARNING models=phaseforge train=stage2 project.seed=42 project.output_dir=/content/PhaseForge/outputs/v2_g3 project.method=phaseforge train.stage1_ckpt_path=/content/PhaseForge/outputs/v2_g3/phaseforge/stage1/seed42/2026-08-19_16-03-41_2db3a926/checkpoints/checkpoint_best.pt train.early_stopping.enabled=false

[2/15] OK phaseforge seed=42 stage2

[runner] $ /content/PhaseForge/.venv/bin/phaseforge-eval project.log_level=WARNING models=phaseforge project.seed=42 project.output_dir=/content/PhaseForge/outputs/v2_g3 train.stage1_ckpt_path=/content/PhaseForge/outputs/v2_g3/phaseforge/stage2/seed42/2026-08-19_16-04-55_d9da6dc6/checkpoints/checkpoint_best.pt eval=rollout eval.mode=rollout project.method=phaseforge train.early_stopping.enabled=false
[robosuite WARNING] No private macro file found! (macros.py:53)
[2026-08-19 16:08:07,567][robosuite_logs][WARNING] - No private macro file found!
[robosuite WARNING] It is recommended to use a private macro file (macros.py:54)
[2026-08-19 16:08:07,567][robosuite_logs][WARNING] - It is recommended to use a private macro file
[robosuite WARNING] To setup, run: python /content/PhaseForge/.venv/lib/python3.12/site-packages/robosuite/scripts/setup_macros.py (macros.py:55)
[2026-08-19 16:08:07,568][robosuite_logs][WARNING] - To setup, run: python /content/PhaseForge/.venv/lib/python3.12/site-packages/robosuite/scripts/setup_macros.py
[robosuite WARNING] Could not import robosuite_models. Some robots may not be available. If you want to use these robots, please install robosuite_models from source (https://github.com/ARISE-Initiative/robosuite_models) or through pip install. (__init__.py:30)
[2026-08-19 16:08:08,037][robosuite_logs][WARNING] - Could not import robosuite_models. Some robots may not be available. If you want to use these robots, please install robosuite_models from source (https://github.com/ARISE-Initiative/robosuite_models) or through pip install.
[robosuite WARNING] Could not load the mink-based whole-body IK. Make sure you install related import properly, otherwise you will not be able to use the default IK controller setting for GR1 robot. (__init__.py:40)
[2026-08-19 16:08:08,037][robosuite_logs][WARNING] - Could not load the mink-based whole-body IK. Make sure you install related import properly, otherwise you will not be able to use the default IK controller setting for GR1 robot.
[3/15] OK phaseforge seed=42 eval

[runner] $ /content/PhaseForge/.venv/bin/phaseforge-train project.log_level=WARNING models=phaseforge train=stage1 project.seed=43 project.output_dir=/content/PhaseForge/outputs/v2_g3 project.method=phaseforge train.early_stopping.enabled=false

[4/15] OK phaseforge seed=43 stage1

[runner] $ /content/PhaseForge/.venv/bin/phaseforge-train project.log_level=WARNING models=phaseforge train=stage2 project.seed=43 project.output_dir=/content/PhaseForge/outputs/v2_g3 project.method=phaseforge train.stage1_ckpt_path=/content/PhaseForge/outputs/v2_g3/phaseforge/stage1/seed43/2026-08-19_16-13-26_8e227bb8/checkpoints/checkpoint_best.pt train.early_stopping.enabled=false

[5/15] OK phaseforge seed=43 stage2

[runner] $ /content/PhaseForge/.venv/bin/phaseforge-eval project.log_level=WARNING models=phaseforge project.seed=43 project.output_dir=/content/PhaseForge/outputs/v2_g3 train.stage1_ckpt_path=/content/PhaseForge/outputs/v2_g3/phaseforge/stage2/seed43/2026-08-19_16-14-40_6e525677/checkpoints/checkpoint_best.pt eval=rollout eval.mode=rollout project.method=phaseforge train.early_stopping.enabled=false
[robosuite WARNING] No private macro file found! (macros.py:53)
[2026-08-19 16:17:52,434][robosuite_logs][WARNING] - No private macro file found!
[robosuite WARNING] It is recommended to use a private macro file (macros.py:54)
[2026-08-19 16:17:52,434][robosuite_logs][WARNING] - It is recommended to use a private macro file
[robosuite WARNING] To setup, run: python /content/PhaseForge/.venv/lib/python3.12/site-packages/robosuite/scripts/setup_macros.py (macros.py:55)
[2026-08-19 16:17:52,434][robosuite_logs][WARNING] - To setup, run: python /content/PhaseForge/.venv/lib/python3.12/site-packages/robosuite/scripts/setup_macros.py
[robosuite WARNING] Could not import robosuite_models. Some robots may not be available. If you want to use these robots, please install robosuite_models from source (https://github.com/ARISE-Initiative/robosuite_models) or through pip install. (__init__.py:30)
[2026-08-19 16:17:52,897][robosuite_logs][WARNING] - Could not import robosuite_models. Some robots may not be available. If you want to use these robots, please install robosuite_models from source (https://github.com/ARISE-Initiative/robosuite_models) or through pip install.
[robosuite WARNING] Could not load the mink-based whole-body IK. Make sure you install related import properly, otherwise you will not be able to use the default IK controller setting for GR1 robot. (__init__.py:40)
[2026-08-19 16:17:52,898][robosuite_logs][WARNING] - Could not load the mink-based whole-body IK. Make sure you install related import properly, otherwise you will not be able to use the default IK controller setting for GR1 robot.
[6/15] OK phaseforge seed=43 eval

[runner] $ /content/PhaseForge/.venv/bin/phaseforge-train project.log_level=WARNING models=phaseforge train=stage1 project.seed=44 project.output_dir=/content/PhaseForge/outputs/v2_g3 project.method=phaseforge train.early_stopping.enabled=false

[7/15] OK phaseforge seed=44 stage1

[runner] $ /content/PhaseForge/.venv/bin/phaseforge-train project.log_level=WARNING models=phaseforge train=stage2 project.seed=44 project.output_dir=/content/PhaseForge/outputs/v2_g3 project.method=phaseforge train.stage1_ckpt_path=/content/PhaseForge/outputs/v2_g3/phaseforge/stage1/seed44/2026-08-19_16-23-40_2b3902d6/checkpoints/checkpoint_best.pt train.early_stopping.enabled=false

[8/15] OK phaseforge seed=44 stage2

[runner] $ /content/PhaseForge/.venv/bin/phaseforge-eval project.log_level=WARNING models=phaseforge project.seed=44 project.output_dir=/content/PhaseForge/outputs/v2_g3 train.stage1_ckpt_path=/content/PhaseForge/outputs/v2_g3/phaseforge/stage2/seed44/2026-08-19_16-24-54_5103f904/checkpoints/checkpoint_best.pt eval=rollout eval.mode=rollout project.method=phaseforge train.early_stopping.enabled=false
[robosuite WARNING] No private macro file found! (macros.py:53)
[2026-08-19 16:28:06,462][robosuite_logs][WARNING] - No private macro file found!
[robosuite WARNING] It is recommended to use a private macro file (macros.py:54)
[2026-08-19 16:28:06,464][robosuite_logs][WARNING] - It is recommended to use a private macro file
[robosuite WARNING] To setup, run: python /content/PhaseForge/.venv/lib/python3.12/site-packages/robosuite/scripts/setup_macros.py (macros.py:55)
[2026-08-19 16:28:06,464][robosuite_logs][WARNING] - To setup, run: python /content/PhaseForge/.venv/lib/python3.12/site-packages/robosuite/scripts/setup_macros.py
[robosuite WARNING] Could not import robosuite_models. Some robots may not be available. If you want to use these robots, please install robosuite_models from source (https://github.com/ARISE-Initiative/robosuite_models) or through pip install. (__init__.py:30)
[2026-08-19 16:28:06,924][robosuite_logs][WARNING] - Could not import robosuite_models. Some robots may not be available. If you want to use these robots, please install robosuite_models from source (https://github.com/ARISE-Initiative/robosuite_models) or through pip install.
[robosuite WARNING] Could not load the mink-based whole-body IK. Make sure you install related import properly, otherwise you will not be able to use the default IK controller setting for GR1 robot. (__init__.py:40)
[2026-08-19 16:28:06,924][robosuite_logs][WARNING] - Could not load the mink-based whole-body IK. Make sure you install related import properly, otherwise you will not be able to use the default IK controller setting for GR1 robot.
[9/15] OK phaseforge seed=44 eval

[runner] $ /content/PhaseForge/.venv/bin/phaseforge-train project.log_level=WARNING models=baselines/bc train=stage1 project.seed=42 project.output_dir=/content/PhaseForge/outputs/v2_g3 project.method=bc train.early_stopping.enabled=false

[10/15] OK bc seed=42 stage1

[runner] $ /content/PhaseForge/.venv/bin/phaseforge-eval project.log_level=WARNING models=baselines/bc project.seed=42 project.output_dir=/content/PhaseForge/outputs/v2_g3 train.stage1_ckpt_path=/content/PhaseForge/outputs/v2_g3/bc/stage1/seed42/2026-08-19_16-34-53_c9b19fb1/checkpoints/checkpoint_best.pt eval=rollout eval.mode=rollout project.method=bc train.early_stopping.enabled=false
[robosuite WARNING] No private macro file found! (macros.py:53)
[2026-08-19 16:36:04,245][robosuite_logs][WARNING] - No private macro file found!
[robosuite WARNING] It is recommended to use a private macro file (macros.py:54)
[2026-08-19 16:36:04,246][robosuite_logs][WARNING] - It is recommended to use a private macro file
[robosuite WARNING] To setup, run: python /content/PhaseForge/.venv/lib/python3.12/site-packages/robosuite/scripts/setup_macros.py (macros.py:55)
[2026-08-19 16:36:04,246][robosuite_logs][WARNING] - To setup, run: python /content/PhaseForge/.venv/lib/python3.12/site-packages/robosuite/scripts/setup_macros.py
[robosuite WARNING] Could not import robosuite_models. Some robots may not be available. If you want to use these robots, please install robosuite_models from source (https://github.com/ARISE-Initiative/robosuite_models) or through pip install. (__init__.py:30)
[2026-08-19 16:36:04,723][robosuite_logs][WARNING] - Could not import robosuite_models. Some robots may not be available. If you want to use these robots, please install robosuite_models from source (https://github.com/ARISE-Initiative/robosuite_models) or through pip install.
[robosuite WARNING] Could not load the mink-based whole-body IK. Make sure you install related import properly, otherwise you will not be able to use the default IK controller setting for GR1 robot. (__init__.py:40)
[2026-08-19 16:36:04,724][robosuite_logs][WARNING] - Could not load the mink-based whole-body IK. Make sure you install related import properly, otherwise you will not be able to use the default IK controller setting for GR1 robot.
[11/15] OK bc seed=42 eval

[runner] $ /content/PhaseForge/.venv/bin/phaseforge-train project.log_level=WARNING models=baselines/bc train=stage1 project.seed=43 project.output_dir=/content/PhaseForge/outputs/v2_g3 project.method=bc train.early_stopping.enabled=false

[12/15] OK bc seed=43 stage1

[runner] $ /content/PhaseForge/.venv/bin/phaseforge-eval project.log_level=WARNING models=baselines/bc project.seed=43 project.output_dir=/content/PhaseForge/outputs/v2_g3 train.stage1_ckpt_path=/content/PhaseForge/outputs/v2_g3/bc/stage1/seed43/2026-08-19_16-40-30_68b5a3ed/checkpoints/checkpoint_best.pt eval=rollout eval.mode=rollout project.method=bc train.early_stopping.enabled=false
[robosuite WARNING] No private macro file found! (macros.py:53)
[2026-08-19 16:41:42,558][robosuite_logs][WARNING] - No private macro file found!
[robosuite WARNING] It is recommended to use a private macro file (macros.py:54)
[2026-08-19 16:41:42,559][robosuite_logs][WARNING] - It is recommended to use a private macro file
[robosuite WARNING] To setup, run: python /content/PhaseForge/.venv/lib/python3.12/site-packages/robosuite/scripts/setup_macros.py (macros.py:55)
[2026-08-19 16:41:42,559][robosuite_logs][WARNING] - To setup, run: python /content/PhaseForge/.venv/lib/python3.12/site-packages/robosuite/scripts/setup_macros.py
[robosuite WARNING] Could not import robosuite_models. Some robots may not be available. If you want to use these robots, please install robosuite_models from source (https://github.com/ARISE-Initiative/robosuite_models) or through pip install. (__init__.py:30)
[2026-08-19 16:41:43,015][robosuite_logs][WARNING] - Could not import robosuite_models. Some robots may not be available. If you want to use these robots, please install robosuite_models from source (https://github.com/ARISE-Initiative/robosuite_models) or through pip install.
[robosuite WARNING] Could not load the mink-based whole-body IK. Make sure you install related import properly, otherwise you will not be able to use the default IK controller setting for GR1 robot. (__init__.py:40)
[2026-08-19 16:41:43,016][robosuite_logs][WARNING] - Could not load the mink-based whole-body IK. Make sure you install related import properly, otherwise you will not be able to use the default IK controller setting for GR1 robot.
[13/15] OK bc seed=43 eval

[runner] $ /content/PhaseForge/.venv/bin/phaseforge-train project.log_level=WARNING models=baselines/bc train=stage1 project.seed=44 project.output_dir=/content/PhaseForge/outputs/v2_g3 project.method=bc train.early_stopping.enabled=false

[14/15] OK bc seed=44 stage1

[runner] $ /content/PhaseForge/.venv/bin/phaseforge-eval project.log_level=WARNING models=baselines/bc project.seed=44 project.output_dir=/content/PhaseForge/outputs/v2_g3 train.stage1_ckpt_path=/content/PhaseForge/outputs/v2_g3/bc/stage1/seed44/2026-08-19_16-47-10_9b7549e3/checkpoints/checkpoint_best.pt eval=rollout eval.mode=rollout project.method=bc train.early_stopping.enabled=false
[robosuite WARNING] No private macro file found! (macros.py:53)
[2026-08-19 16:48:26,369][robosuite_logs][WARNING] - No private macro file found!
[robosuite WARNING] It is recommended to use a private macro file (macros.py:54)
[2026-08-19 16:48:26,369][robosuite_logs][WARNING] - It is recommended to use a private macro file
[robosuite WARNING] To setup, run: python /content/PhaseForge/.venv/lib/python3.12/site-packages/robosuite/scripts/setup_macros.py (macros.py:55)
[2026-08-19 16:48:26,369][robosuite_logs][WARNING] - To setup, run: python /content/PhaseForge/.venv/lib/python3.12/site-packages/robosuite/scripts/setup_macros.py
[robosuite WARNING] Could not import robosuite_models. Some robots may not be available. If you want to use these robots, please install robosuite_models from source (https://github.com/ARISE-Initiative/robosuite_models) or through pip install. (__init__.py:30)
[2026-08-19 16:48:26,863][robosuite_logs][WARNING] - Could not import robosuite_models. Some robots may not be available. If you want to use these robots, please install robosuite_models from source (https://github.com/ARISE-Initiative/robosuite_models) or through pip install.
[robosuite WARNING] Could not load the mink-based whole-body IK. Make sure you install related import properly, otherwise you will not be able to use the default IK controller setting for GR1 robot. (__init__.py:40)
[2026-08-19 16:48:26,863][robosuite_logs][WARNING] - Could not load the mink-based whole-body IK. Make sure you install related import properly, otherwise you will not be able to use the default IK controller setting for GR1 robot.
[15/15] OK bc seed=44 eval

[runner] summary
  ran=15 skipped=0 failed=0
  phaseforge                       seed42=[ok ok ok]  seed43=[ok ok ok]  seed44=[ok ok ok]
  bc                               seed42=[ok ok]  seed43=[ok ok]  seed44=[ok ok]
  bc_large                         seed42=[- -]  seed43=[- -]  seed44=[- -]
  bc_robot_only                    seed42=[- -]  seed43=[- -]  seed44=[- -]
  scratch_moe                      seed42=[- -]  seed43=[- -]  seed44=[- -]
  warmstart_moe                    seed42=[- -]  seed43=[- -]  seed44=[- -]
  phase_pretrain_random_router     seed42=[- -]  seed43=[- -]  seed44=[- -]
  plain_encoder_phase_bootstrap    seed42=[- -]  seed43=[- -]  seed44=[- -]
  pf_spherical_kmeans              seed42=[- -]  seed43=[- -]  seed44=[- -]
  pf_kmeans                        seed42=[- -]  seed43=[- -]  seed44=[- -]
  pf_phase_head                    seed42=[- -]  seed43=[- -]  seed44=[- -]
  pf_random_random                 seed42=[- -]  seed43=[- -]  seed44=[- -]
  pf_centroid_random               seed42=[- -]  seed43=[- -]  seed44=[- -]
  pf_spherical                     seed42=[- -]  seed43=[- -]  seed44=[- -]
  pf_ft                            seed42=[- -]  seed43=[- -]  seed44=[- -]
  teacher_forced                   seed42=[- -]  seed43=[- -]  seed44=[- -]
  pf_k3                            seed42=[- -]  seed43=[- -]  seed44=[- -]
  pf_k12                           seed42=[- -]  seed43=[- -]  seed44=[- -]
  pf_jitter_00                     seed42=[- -]  seed43=[- -]  seed44=[- -]
  pf_jitter_10                     seed42=[- -]  seed43=[- -]  seed44=[- -]
  pf_corrupt_25                    seed42=[- -]  seed43=[- -]  seed44=[- -]
  pf_corrupt_50                    seed42=[- -]  seed43=[- -]  seed44=[- -]
  pf_shuffle_control               seed42=[- -]  seed43=[- -]  seed44=[- -]
  pf_random_warm                   seed42=[- -]  seed43=[- -]  seed44=[- -]
[v2-gates] g3: findings -> outputs/_findings/v2_gates_g3.json
  bc:42: SR=0.6 [0.4618118910275937,0.7239181021375425] bank=a7d3953c0afcf560
  bc:43: SR=0.48 [0.3479691239550571,0.6148848774119157] bank=a7d3953c0afcf560
  bc:44: SR=0.54 [0.40398634328388083,0.6703056539821737] bank=a7d3953c0afcf560
  phaseforge:42: SR=0.56 [0.4230578680014353,0.6883801278976466] bank=a7d3953c0afcf560
  phaseforge:43: SR=0.48 [0.3479691239550571,0.6148848774119157] bank=a7d3953c0afcf560
  phaseforge:44: SR=0.38 [0.2586259584257722,0.5184980497760643] bank=a7d3953c0afcf560


[g2] cache: 200 trajectories, 8640 train states

[g2] per-phase support:
  phase 5 (approach): n=1470, dispersion=34.4096
  phase 5 (pre-grasp): n=100, dispersion=19.3200
  phase 5 (grasp): n=2249, dispersion=41.7776
  phase 5 (transport): n=3817, dispersion=49.5528
  phase 5 (place): n=955, dispersion=60.5869
  phase 5 (retract): n=49, dispersion=50.7724

[g2] pairwise confusions (centroid classifier):
  0->1: 0.376
  0->2: 0.327
  0->3: 0.295
  0->4: 0.003
  0->5: 0.000
  1->0: 0.000
  1->2: 0.230
  1->3: 0.060
  1->4: 0.000
  1->5: 0.000
  2->0: 0.078
  2->1: 0.335
  2->3: 0.241
  2->4: 0.064
  2->5: 0.028
  3->0: 0.132
  3->1: 0.277
  3->2: 0.543
  3->4: 0.120
  3->5: 0.046
  4->0: 0.067
  4->1: 0.000
  4->2: 0.002
  4->3: 0.002
  4->5: 0.237
  5->0: 0.000
  5->1: 0.000
  5->2: 0.000
  5->3: 0.000
  5->4: 0.041

[g2] Cohen's d (centroid separation):
  0|1: 8.646
  0|2: 6.499
  0|3: 5.780
  0|4: 7.291
  0|5: 11.376
  1|2: 2.831
  1|3: 3.860
  1|4: 11.734
  1|5: 16.383
  2|3: 1.309
  2|4: 9.513
  2|5: 12.996
  3|4: 8.283
  3|5: 11.496
  4|5: 3.890

[g2] merged schemes:
  6_phases: mean_confusion=0.117, silhouette=0.129
  5_merge_grasp_transport: mean_confusion=0.087, silhouette=0.178
  4_super: mean_confusion=0.078, silhouette=0.224

[g2] VERDICT: merge 2+3 recommended = True (conf 2<->3 = 0.543, conf 1<->5 = 0.000)
[g2] written to outputs/_findings/phase_merge_separability.json


[v2-gates] g1: teacher_forced re-check under V2-B (EXP-116)
[v2-gates] g1: phaseforge-sweep --manifest /content/PhaseForge/experiments/lift_ablation.json --methods teacher_forced --seeds 42,43,44 --outputs outputs/v2_g1
[runner] commit gate: 446e947

[runner] plan (6 steps, outputs base: /content/PhaseForge/outputs/v2_g1)
    1. teacher_forced seed=42 stage2                    pending
    2. teacher_forced seed=42 eval                      pending
    3. teacher_forced seed=43 stage2                    pending
    4. teacher_forced seed=43 eval                      pending
    5. teacher_forced seed=44 stage2                    pending
    6. teacher_forced seed=44 eval                      pending

[runner] auto-injecting missing dependency: phaseforge seed=42 stage1

[runner] $ /content/PhaseForge/.venv/bin/phaseforge-train project.log_level=WARNING models=phaseforge train=stage1 project.seed=42 project.output_dir=/content/PhaseForge/outputs/v2_g1 project.method=phaseforge train.early_stopping.enabled=false


[runner] $ /content/PhaseForge/.venv/bin/phaseforge-train project.log_level=WARNING models=baselines/teacher_forced train=stage2 project.seed=42 project.output_dir=/content/PhaseForge/outputs/v2_g1 project.method=teacher_forced train.stage1_ckpt_path=/content/PhaseForge/outputs/v2_g1/phaseforge/stage1/seed42/2026-08-19_15-31-17_df7b0d75/checkpoints/checkpoint_best.pt train.early_stopping.enabled=false

[1/6] OK teacher_forced seed=42 stage2

[runner] $ /content/PhaseForge/.venv/bin/phaseforge-eval project.log_level=WARNING models=baselines/teacher_forced project.seed=42 project.output_dir=/content/PhaseForge/outputs/v2_g1 train.stage1_ckpt_path=/content/PhaseForge/outputs/v2_g1/teacher_forced/stage2/seed42/2026-08-19_15-32-30_958f9096/checkpoints/checkpoint_best.pt eval=rollout eval.mode=rollout project.method=teacher_forced train.early_stopping.enabled=false
[2026-08-19 15:35:09,553][phaseforge.evaluations.rollout.runner][WARNING] - Reset bank a7d3953c0afcf560 does not exist — generating it now (one-time artifact, then frozen and verified on every load).
/content/PhaseForge/.venv/lib/python3.12/site-packages/robosuite/__init__.py:48: SyntaxWarning: invalid escape sequence '\ '
  /[_]\  [~]\/    |//  |
[robosuite WARNING] No private macro file found! (macros.py:53)
[2026-08-19 15:35:09,901][robosuite_logs][WARNING] - No private macro file found!
[robosuite WARNING] It is recommended to use a private macro file (macros.py:54)
[2026-08-19 15:35:09,901][robosuite_logs][WARNING] - It is recommended to use a private macro file
[robosuite WARNING] To setup, run: python /content/PhaseForge/.venv/lib/python3.12/site-packages/robosuite/scripts/setup_macros.py (macros.py:55)
[2026-08-19 15:35:09,901][robosuite_logs][WARNING] - To setup, run: python /content/PhaseForge/.venv/lib/python3.12/site-packages/robosuite/scripts/setup_macros.py
/content/PhaseForge/.venv/lib/python3.12/site-packages/robosuite/models/robots/robot_model.py:147: SyntaxWarning: invalid escape sequence '\s'
  Throws error if robot already has a mount or if mount type i\s incorrect.
[robosuite WARNING] Could not import robosuite_models. Some robots may not be available. If you want to use these robots, please install robosuite_models from source (https://github.com/ARISE-Initiative/robosuite_models) or through pip install. (__init__.py:30)
[2026-08-19 15:35:12,443][robosuite_logs][WARNING] - Could not import robosuite_models. Some robots may not be available. If you want to use these robots, please install robosuite_models from source (https://github.com/ARISE-Initiative/robosuite_models) or through pip install.
[robosuite WARNING] Could not load the mink-based whole-body IK. Make sure you install related import properly, otherwise you will not be able to use the default IK controller setting for GR1 robot. (__init__.py:40)
[2026-08-19 15:35:12,444][robosuite_logs][WARNING] - Could not load the mink-based whole-body IK. Make sure you install related import properly, otherwise you will not be able to use the default IK controller setting for GR1 robot.
[2/6] OK teacher_forced seed=42 eval

[runner] auto-injecting missing dependency: phaseforge seed=43 stage1

[runner] $ /content/PhaseForge/.venv/bin/phaseforge-train project.log_level=WARNING models=phaseforge train=stage1 project.seed=43 project.output_dir=/content/PhaseForge/outputs/v2_g1 project.method=phaseforge train.early_stopping.enabled=false


[runner] $ /content/PhaseForge/.venv/bin/phaseforge-train project.log_level=WARNING models=baselines/teacher_forced train=stage2 project.seed=43 project.output_dir=/content/PhaseForge/outputs/v2_g1 project.method=teacher_forced train.stage1_ckpt_path=/content/PhaseForge/outputs/v2_g1/phaseforge/stage1/seed43/2026-08-19_15-41-16_803b4ca7/checkpoints/checkpoint_best.pt train.early_stopping.enabled=false

[3/6] OK teacher_forced seed=43 stage2

[runner] $ /content/PhaseForge/.venv/bin/phaseforge-eval project.log_level=WARNING models=baselines/teacher_forced project.seed=43 project.output_dir=/content/PhaseForge/outputs/v2_g1 train.stage1_ckpt_path=/content/PhaseForge/outputs/v2_g1/teacher_forced/stage2/seed43/2026-08-19_15-42-30_defaa3c8/checkpoints/checkpoint_best.pt eval=rollout eval.mode=rollout project.method=teacher_forced train.early_stopping.enabled=false
[robosuite WARNING] No private macro file found! (macros.py:53)
[2026-08-19 15:45:08,260][robosuite_logs][WARNING] - No private macro file found!
[robosuite WARNING] It is recommended to use a private macro file (macros.py:54)
[2026-08-19 15:45:08,261][robosuite_logs][WARNING] - It is recommended to use a private macro file
[robosuite WARNING] To setup, run: python /content/PhaseForge/.venv/lib/python3.12/site-packages/robosuite/scripts/setup_macros.py (macros.py:55)
[2026-08-19 15:45:08,261][robosuite_logs][WARNING] - To setup, run: python /content/PhaseForge/.venv/lib/python3.12/site-packages/robosuite/scripts/setup_macros.py
[robosuite WARNING] Could not import robosuite_models. Some robots may not be available. If you want to use these robots, please install robosuite_models from source (https://github.com/ARISE-Initiative/robosuite_models) or through pip install. (__init__.py:30)
[2026-08-19 15:45:08,714][robosuite_logs][WARNING] - Could not import robosuite_models. Some robots may not be available. If you want to use these robots, please install robosuite_models from source (https://github.com/ARISE-Initiative/robosuite_models) or through pip install.
[robosuite WARNING] Could not load the mink-based whole-body IK. Make sure you install related import properly, otherwise you will not be able to use the default IK controller setting for GR1 robot. (__init__.py:40)
[2026-08-19 15:45:08,714][robosuite_logs][WARNING] - Could not load the mink-based whole-body IK. Make sure you install related import properly, otherwise you will not be able to use the default IK controller setting for GR1 robot.
[4/6] OK teacher_forced seed=43 eval

[runner] auto-injecting missing dependency: phaseforge seed=44 stage1

[runner] $ /content/PhaseForge/.venv/bin/phaseforge-train project.log_level=WARNING models=phaseforge train=stage1 project.seed=44 project.output_dir=/content/PhaseForge/outputs/v2_g1 project.method=phaseforge train.early_stopping.enabled=false


[runner] $ /content/PhaseForge/.venv/bin/phaseforge-train project.log_level=WARNING models=baselines/teacher_forced train=stage2 project.seed=44 project.output_dir=/content/PhaseForge/outputs/v2_g1 project.method=teacher_forced train.stage1_ckpt_path=/content/PhaseForge/outputs/v2_g1/phaseforge/stage1/seed44/2026-08-19_15-49-20_cdcfba21/checkpoints/checkpoint_best.pt train.early_stopping.enabled=false

[5/6] OK teacher_forced seed=44 stage2

[runner] $ /content/PhaseForge/.venv/bin/phaseforge-eval project.log_level=WARNING models=baselines/teacher_forced project.seed=44 project.output_dir=/content/PhaseForge/outputs/v2_g1 train.stage1_ckpt_path=/content/PhaseForge/outputs/v2_g1/teacher_forced/stage2/seed44/2026-08-19_15-50-33_f0f51d97/checkpoints/checkpoint_best.pt eval=rollout eval.mode=rollout project.method=teacher_forced train.early_stopping.enabled=false
[robosuite WARNING] No private macro file found! (macros.py:53)
[2026-08-19 15:53:11,853][robosuite_logs][WARNING] - No private macro file found!
[robosuite WARNING] It is recommended to use a private macro file (macros.py:54)
[2026-08-19 15:53:11,854][robosuite_logs][WARNING] - It is recommended to use a private macro file
[robosuite WARNING] To setup, run: python /content/PhaseForge/.venv/lib/python3.12/site-packages/robosuite/scripts/setup_macros.py (macros.py:55)
[2026-08-19 15:53:11,854][robosuite_logs][WARNING] - To setup, run: python /content/PhaseForge/.venv/lib/python3.12/site-packages/robosuite/scripts/setup_macros.py
[robosuite WARNING] Could not import robosuite_models. Some robots may not be available. If you want to use these robots, please install robosuite_models from source (https://github.com/ARISE-Initiative/robosuite_models) or through pip install. (__init__.py:30)
[2026-08-19 15:53:12,449][robosuite_logs][WARNING] - Could not import robosuite_models. Some robots may not be available. If you want to use these robots, please install robosuite_models from source (https://github.com/ARISE-Initiative/robosuite_models) or through pip install.
[robosuite WARNING] Could not load the mink-based whole-body IK. Make sure you install related import properly, otherwise you will not be able to use the default IK controller setting for GR1 robot. (__init__.py:40)
[2026-08-19 15:53:12,449][robosuite_logs][WARNING] - Could not load the mink-based whole-body IK. Make sure you install related import properly, otherwise you will not be able to use the default IK controller setting for GR1 robot.
[6/6] OK teacher_forced seed=44 eval

[runner] summary
  ran=6 skipped=0 failed=0
  phaseforge                       seed42=[ok - -]  seed43=[ok - -]  seed44=[ok - -]
  bc                               seed42=[- -]  seed43=[- -]  seed44=[- -]
  bc_large                         seed42=[- -]  seed43=[- -]  seed44=[- -]
  bc_robot_only                    seed42=[- -]  seed43=[- -]  seed44=[- -]
  scratch_moe                      seed42=[- -]  seed43=[- -]  seed44=[- -]
  warmstart_moe                    seed42=[- -]  seed43=[- -]  seed44=[- -]
  phase_pretrain_random_router     seed42=[- -]  seed43=[- -]  seed44=[- -]
  plain_encoder_phase_bootstrap    seed42=[- -]  seed43=[- -]  seed44=[- -]
  pf_spherical_kmeans              seed42=[- -]  seed43=[- -]  seed44=[- -]
  pf_kmeans                        seed42=[- -]  seed43=[- -]  seed44=[- -]
  pf_phase_head                    seed42=[- -]  seed43=[- -]  seed44=[- -]
  pf_random_random                 seed42=[- -]  seed43=[- -]  seed44=[- -]
  pf_centroid_random               seed42=[- -]  seed43=[- -]  seed44=[- -]
  pf_spherical                     seed42=[- -]  seed43=[- -]  seed44=[- -]
  pf_ft                            seed42=[- -]  seed43=[- -]  seed44=[- -]
  teacher_forced                   seed42=[ok ok]  seed43=[ok ok]  seed44=[ok ok]
  pf_k3                            seed42=[- -]  seed43=[- -]  seed44=[- -]
  pf_k12                           seed42=[- -]  seed43=[- -]  seed44=[- -]
  pf_jitter_00                     seed42=[- -]  seed43=[- -]  seed44=[- -]
  pf_jitter_10                     seed42=[- -]  seed43=[- -]  seed44=[- -]
  pf_corrupt_25                    seed42=[- -]  seed43=[- -]  seed44=[- -]
  pf_corrupt_50                    seed42=[- -]  seed43=[- -]  seed44=[- -]
  pf_shuffle_control               seed42=[- -]  seed43=[- -]  seed44=[- -]
  pf_random_warm                   seed42=[- -]  seed43=[- -]  seed44=[- -]
[v2-gates] g1: findings -> outputs/_findings/v2_gates_g1.json
  teacher_forced:42: SR=0.52 [0.3851151225880844,0.6520308760449429] bank=a7d3953c0afcf560
  teacher_forced:43: SR=0.66 [0.5215356854148285,0.7756323036493895] bank=a7d3953c0afcf560
  teacher_forced:44: SR=0.4 [0.2760818978624575,0.5381881089724063] bank=a7d3953c0afcf560

==

[v2-gates] g6: E=6 centroid re-run on shared bank (EXP-209)
[v2-gates] g6: phaseforge-sweep --manifest /content/PhaseForge/experiments/lift_ablation.json --methods phaseforge_e6 --seeds 42,43,44 --outputs outputs/v2_e6_diag
[runner] commit gate: 1725952

[runner] plan (9 steps, outputs base: /content/PhaseForge/outputs/v2_e6_diag)
    1. phaseforge_e6 seed=42 stage1                     pending
    2. phaseforge_e6 seed=42 stage2                     pending
    3. phaseforge_e6 seed=42 eval                       pending
    4. phaseforge_e6 seed=43 stage1                     pending
    5. phaseforge_e6 seed=43 stage2                     pending
    6. phaseforge_e6 seed=43 eval                       pending
    7. phaseforge_e6 seed=44 stage1                     pending
    8. phaseforge_e6 seed=44 stage2                     pending
    9. phaseforge_e6 seed=44 eval                       pending

[runner] $ /content/PhaseForge/.venv/bin/phaseforge-train project.log_level=WARNING models=phaseforge train=stage1 project.seed=42 project.output_dir=/content/PhaseForge/outputs/v2_e6_diag project.tag=e6_diag project.method=phaseforge_e6 train.early_stopping.enabled=false models.router.num_experts=6 models.router_init.type=centroid

[1/9] OK phaseforge_e6 seed=42 stage1

[runner] $ /content/PhaseForge/.venv/bin/phaseforge-train project.log_level=WARNING models=phaseforge train=stage2 project.seed=42 project.output_dir=/content/PhaseForge/outputs/v2_e6_diag project.tag=e6_diag project.method=phaseforge_e6 train.stage1_ckpt_path=/content/PhaseForge/outputs/v2_e6_diag/phaseforge/stage1/seed42/2026-08-19_22-13-13_e6_diag_7930b4f3/checkpoints/checkpoint_best.pt train.early_stopping.enabled=false models.router.num_experts=6 models.router_init.type=centroid

[2/9] OK phaseforge_e6 seed=42 stage2

[runner] $ /content/PhaseForge/.venv/bin/phaseforge-eval project.log_level=WARNING models=phaseforge project.seed=42 project.output_dir=/content/PhaseForge/outputs/v2_e6_diag train.stage1_ckpt_path=/content/PhaseForge/outputs/v2_e6_diag/phaseforge/stage2/seed42/2026-08-19_22-14-42_e6_diag_e9333207/checkpoints/checkpoint_best.pt eval=rollout eval.mode=rollout project.tag=e6_diag project.method=phaseforge_e6 train.early_stopping.enabled=false models.router.num_experts=6 models.router_init.type=centroid
[2026-08-19 22:17:55,382][phaseforge.evaluations.rollout.runner][WARNING] - Reset bank a7d3953c0afcf560 does not exist — generating it now (one-time artifact, then frozen and verified on every load).
/content/PhaseForge/.venv/lib/python3.12/site-packages/robosuite/__init__.py:48: SyntaxWarning: invalid escape sequence '\ '
  /[_]\  [~]\/    |//  |
[robosuite WARNING] No private macro file found! (macros.py:53)
[2026-08-19 22:17:55,739][robosuite_logs][WARNING] - No private macro file found!
[robosuite WARNING] It is recommended to use a private macro file (macros.py:54)
[2026-08-19 22:17:55,739][robosuite_logs][WARNING] - It is recommended to use a private macro file
[robosuite WARNING] To setup, run: python /content/PhaseForge/.venv/lib/python3.12/site-packages/robosuite/scripts/setup_macros.py (macros.py:55)
[2026-08-19 22:17:55,739][robosuite_logs][WARNING] - To setup, run: python /content/PhaseForge/.venv/lib/python3.12/site-packages/robosuite/scripts/setup_macros.py
/content/PhaseForge/.venv/lib/python3.12/site-packages/robosuite/models/robots/robot_model.py:147: SyntaxWarning: invalid escape sequence '\s'
  Throws error if robot already has a mount or if mount type i\s incorrect.
[robosuite WARNING] Could not import robosuite_models. Some robots may not be available. If you want to use these robots, please install robosuite_models from source (https://github.com/ARISE-Initiative/robosuite_models) or through pip install. (__init__.py:30)
[2026-08-19 22:17:58,060][robosuite_logs][WARNING] - Could not import robosuite_models. Some robots may not be available. If you want to use these robots, please install robosuite_models from source (https://github.com/ARISE-Initiative/robosuite_models) or through pip install.
[robosuite WARNING] Could not load the mink-based whole-body IK. Make sure you install related import properly, otherwise you will not be able to use the default IK controller setting for GR1 robot. (__init__.py:40)
[2026-08-19 22:17:58,061][robosuite_logs][WARNING] - Could not load the mink-based whole-body IK. Make sure you install related import properly, otherwise you will not be able to use the default IK controller setting for GR1 robot.
[3/9] OK phaseforge_e6 seed=42 eval

[runner] $ /content/PhaseForge/.venv/bin/phaseforge-train project.log_level=WARNING models=phaseforge train=stage1 project.seed=43 project.output_dir=/content/PhaseForge/outputs/v2_e6_diag project.tag=e6_diag project.method=phaseforge_e6 train.early_stopping.enabled=false models.router.num_experts=6 models.router_init.type=centroid

[4/9] OK phaseforge_e6 seed=43 stage1

[runner] $ /content/PhaseForge/.venv/bin/phaseforge-train project.log_level=WARNING models=phaseforge train=stage2 project.seed=43 project.output_dir=/content/PhaseForge/outputs/v2_e6_diag project.tag=e6_diag project.method=phaseforge_e6 train.stage1_ckpt_path=/content/PhaseForge/outputs/v2_e6_diag/phaseforge/stage1/seed43/2026-08-19_22-23-11_e6_diag_bc69746f/checkpoints/checkpoint_best.pt train.early_stopping.enabled=false models.router.num_experts=6 models.router_init.type=centroid

[5/9] OK phaseforge_e6 seed=43 stage2

[runner] $ /content/PhaseForge/.venv/bin/phaseforge-eval project.log_level=WARNING models=phaseforge project.seed=43 project.output_dir=/content/PhaseForge/outputs/v2_e6_diag train.stage1_ckpt_path=/content/PhaseForge/outputs/v2_e6_diag/phaseforge/stage2/seed43/2026-08-19_22-24-29_e6_diag_b6389f72/checkpoints/checkpoint_best.pt eval=rollout eval.mode=rollout project.tag=e6_diag project.method=phaseforge_e6 train.early_stopping.enabled=false models.router.num_experts=6 models.router_init.type=centroid
[robosuite WARNING] No private macro file found! (macros.py:53)
[2026-08-19 22:27:40,980][robosuite_logs][WARNING] - No private macro file found!
[robosuite WARNING] It is recommended to use a private macro file (macros.py:54)
[2026-08-19 22:27:40,980][robosuite_logs][WARNING] - It is recommended to use a private macro file
[robosuite WARNING] To setup, run: python /content/PhaseForge/.venv/lib/python3.12/site-packages/robosuite/scripts/setup_macros.py (macros.py:55)
[2026-08-19 22:27:40,980][robosuite_logs][WARNING] - To setup, run: python /content/PhaseForge/.venv/lib/python3.12/site-packages/robosuite/scripts/setup_macros.py
[robosuite WARNING] Could not import robosuite_models. Some robots may not be available. If you want to use these robots, please install robosuite_models from source (https://github.com/ARISE-Initiative/robosuite_models) or through pip install. (__init__.py:30)
[2026-08-19 22:27:41,473][robosuite_logs][WARNING] - Could not import robosuite_models. Some robots may not be available. If you want to use these robots, please install robosuite_models from source (https://github.com/ARISE-Initiative/robosuite_models) or through pip install.
[robosuite WARNING] Could not load the mink-based whole-body IK. Make sure you install related import properly, otherwise you will not be able to use the default IK controller setting for GR1 robot. (__init__.py:40)
[2026-08-19 22:27:41,474][robosuite_logs][WARNING] - Could not load the mink-based whole-body IK. Make sure you install related import properly, otherwise you will not be able to use the default IK controller setting for GR1 robot.
[6/9] OK phaseforge_e6 seed=43 eval

[runner] $ /content/PhaseForge/.venv/bin/phaseforge-train project.log_level=WARNING models=phaseforge train=stage1 project.seed=44 project.output_dir=/content/PhaseForge/outputs/v2_e6_diag project.tag=e6_diag project.method=phaseforge_e6 train.early_stopping.enabled=false models.router.num_experts=6 models.router_init.type=centroid

[7/9] OK phaseforge_e6 seed=44 stage1

[runner] $ /content/PhaseForge/.venv/bin/phaseforge-train project.log_level=WARNING models=phaseforge train=stage2 project.seed=44 project.output_dir=/content/PhaseForge/outputs/v2_e6_diag project.tag=e6_diag project.method=phaseforge_e6 train.stage1_ckpt_path=/content/PhaseForge/outputs/v2_e6_diag/phaseforge/stage1/seed44/2026-08-19_22-31-34_e6_diag_e3121fc1/checkpoints/checkpoint_best.pt train.early_stopping.enabled=false models.router.num_experts=6 models.router_init.type=centroid

[8/9] OK phaseforge_e6 seed=44 stage2

[runner] $ /content/PhaseForge/.venv/bin/phaseforge-eval project.log_level=WARNING models=phaseforge project.seed=44 project.output_dir=/content/PhaseForge/outputs/v2_e6_diag train.stage1_ckpt_path=/content/PhaseForge/outputs/v2_e6_diag/phaseforge/stage2/seed44/2026-08-19_22-32-54_e6_diag_1a5a82cd/checkpoints/checkpoint_best.pt eval=rollout eval.mode=rollout project.tag=e6_diag project.method=phaseforge_e6 train.early_stopping.enabled=false models.router.num_experts=6 models.router_init.type=centroid
[robosuite WARNING] No private macro file found! (macros.py:53)
[2026-08-19 22:36:06,591][robosuite_logs][WARNING] - No private macro file found!
[robosuite WARNING] It is recommended to use a private macro file (macros.py:54)
[2026-08-19 22:36:06,591][robosuite_logs][WARNING] - It is recommended to use a private macro file
[robosuite WARNING] To setup, run: python /content/PhaseForge/.venv/lib/python3.12/site-packages/robosuite/scripts/setup_macros.py (macros.py:55)
[2026-08-19 22:36:06,591][robosuite_logs][WARNING] - To setup, run: python /content/PhaseForge/.venv/lib/python3.12/site-packages/robosuite/scripts/setup_macros.py
[robosuite WARNING] Could not import robosuite_models. Some robots may not be available. If you want to use these robots, please install robosuite_models from source (https://github.com/ARISE-Initiative/robosuite_models) or through pip install. (__init__.py:30)
[2026-08-19 22:36:07,116][robosuite_logs][WARNING] - Could not import robosuite_models. Some robots may not be available. If you want to use these robots, please install robosuite_models from source (https://github.com/ARISE-Initiative/robosuite_models) or through pip install.
[robosuite WARNING] Could not load the mink-based whole-body IK. Make sure you install related import properly, otherwise you will not be able to use the default IK controller setting for GR1 robot. (__init__.py:40)
[2026-08-19 22:36:07,116][robosuite_logs][WARNING] - Could not load the mink-based whole-body IK. Make sure you install related import properly, otherwise you will not be able to use the default IK controller setting for GR1 robot.
[9/9] OK phaseforge_e6 seed=44 eval

[runner] summary
  ran=9 skipped=0 failed=0
  phaseforge                       seed42=[- - -]  seed43=[- - -]  seed44=[- - -]
  bc                               seed42=[- -]  seed43=[- -]  seed44=[- -]
  bc_large                         seed42=[- -]  seed43=[- -]  seed44=[- -]
  bc_robot_only                    seed42=[- -]  seed43=[- -]  seed44=[- -]
  scratch_moe                      seed42=[- -]  seed43=[- -]  seed44=[- -]
  warmstart_moe                    seed42=[- -]  seed43=[- -]  seed44=[- -]
  phase_pretrain_random_router     seed42=[- -]  seed43=[- -]  seed44=[- -]
  plain_encoder_phase_bootstrap    seed42=[- -]  seed43=[- -]  seed44=[- -]
  pf_spherical_kmeans              seed42=[- -]  seed43=[- -]  seed44=[- -]
  pf_kmeans                        seed42=[- -]  seed43=[- -]  seed44=[- -]
  pf_phase_head                    seed42=[- -]  seed43=[- -]  seed44=[- -]
  pf_random_random                 seed42=[- -]  seed43=[- -]  seed44=[- -]
  pf_centroid_random               seed42=[- -]  seed43=[- -]  seed44=[- -]
  pf_spherical                     seed42=[- -]  seed43=[- -]  seed44=[- -]
  pf_ft                            seed42=[- -]  seed43=[- -]  seed44=[- -]
  teacher_forced                   seed42=[- -]  seed43=[- -]  seed44=[- -]
  pf_k3                            seed42=[- -]  seed43=[- -]  seed44=[- -]
  pf_k12                           seed42=[- -]  seed43=[- -]  seed44=[- -]
  pf_jitter_00                     seed42=[- -]  seed43=[- -]  seed44=[- -]
  pf_jitter_10                     seed42=[- -]  seed43=[- -]  seed44=[- -]
  pf_corrupt_25                    seed42=[- -]  seed43=[- -]  seed44=[- -]
  pf_corrupt_50                    seed42=[- -]  seed43=[- -]  seed44=[- -]
  pf_shuffle_control               seed42=[- -]  seed43=[- -]  seed44=[- -]
  pf_random_warm                   seed42=[- -]  seed43=[- -]  seed44=[- -]
  phaseforge_e6                    seed42=[ok ok ok]  seed43=[ok ok ok]  seed44=[ok ok ok]
[v2-gates] g6: findings -> outputs/_findings/v2_gates_g6.json
  phaseforge_e6:42: SR=0.68 [0.5418944286372394,0.792419559060006] bank=a7d3953c0afcf560
  phaseforge_e6:43: SR=0.74 [0.6044657801100008,0.8412862034863262] bank=a7d3953c0afcf560
  phaseforge_e6:44: SR=0.5 [0.36644286412332855,0.6335571358766714] bank=a7d3953c0afcf560