[robosuite WARNING] No private macro file found! (macros.py:53)
[robosuite WARNING] It is recommended to use a private macro file (macros.py:54)
[robosuite WARNING] To setup, run: python /content/PhaseForge/.venv/lib/python3.12/site-packages/robosuite/scripts/setup_macros.py (macros.py:55)
[robosuite WARNING] Could not import robosuite_models. Some robots may not be available. If you want to use these robots, please install robosuite_models from source (https://github.com/ARISE-Initiative/robosuite_models) or through pip install. (__init__.py:30)
[robosuite WARNING] Could not load the mink-based whole-body IK. Make sure you install related import properly, otherwise you will not be able to use the default IK controller setting for GR1 robot. (__init__.py:40)
Task: Lift; bank: a7d3953c0afcf560; cases: [0, 1]
Controller metadata: {
  "robot0.right": {
    "__class__": "OperationalSpaceController",
    "input_type": "delta",
    "input_ref_frame": "world",
    "control_delta": null,
    "input_min": [
      -1.0,
      -1.0,
      -1.0,
      -1.0,
      -1.0,
      -1.0
    ],
    "input_max": [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    "output_min": [
      -0.05,
      -0.05,
      -0.05,
      -0.5,
      -0.5,
      -0.5
    ],
    "output_max": [
      0.05,
      0.05,
      0.05,
      0.5,
      0.5,
      0.5
    ]
  },
  "robot0.right_gripper": {
    "__class__": "SimpleGripController",
    "input_type": null,
    "input_ref_frame": null,
    "control_delta": null,
    "input_min": [
      -1.0,
      -1.0
    ],
    "input_max": [
      1.0,
      1.0
    ],
    "output_min": [
      -1.0,
      -1.0
    ],
    "output_max": [
      1.0,
      1.0
    ]
  }
}
case=00 t=000 phase=APPROACH->APPROACH eef=[-0.10954, -0.00568, 1.01141] obj=[0.01095, 0.02276, 0.82056] a_xyz=[1.0, 0.696, 1.0] grip=1.0 grasp=False success=False
case=00 t=010 phase=APPROACH->DESCEND eef=[-0.00321, 0.0242, 1.09073] obj=[0.01095, 0.02276, 0.81988] a_xyz=[0.471, -0.029, -1.0] grip=1.0 grasp=False success=False
case=00 t=020 phase=GRASP->GRASP eef=[0.01327, 0.02301, 1.00493] obj=[0.01095, 0.02276, 0.81989] a_xyz=[-0.068, -0.006, -0.16] grip=-1.0 grasp=False success=False
case=00 t=030 phase=GRASP->GRASP eef=[0.01048, 0.02296, 0.99963] obj=[0.01095, 0.02276, 0.81989] a_xyz=[0.011, -0.004, 0.006] grip=-1.0 grasp=False success=False
case=00 t=040 phase=GRASP->LIFT eef=[0.01078, 0.02291, 0.99984] obj=[0.01095, 0.02276, 0.81989] a_xyz=[0.004, -0.003, 0.001] grip=-1.0 grasp=False success=False
case=00 t=050 phase=LIFT->LIFT eef=[0.00378, 0.02315, 1.1161] obj=[0.01095, 0.02276, 0.81989] a_xyz=[0.0, 0.0, 0.925] grip=-1.0 grasp=False success=False
case=00 t=060 phase=LIFT->LIFT eef=[0.00379, 0.02323, 1.15026] obj=[0.01095, 0.02276, 0.81989] a_xyz=[0.0, 0.0, -0.002] grip=-1.0 grasp=False success=False
case=00 t=070 phase=LIFT->LIFT eef=[0.00386, 0.02314, 1.15008] obj=[0.01095, 0.02276, 0.81989] a_xyz=[0.0, 0.0, -0.002] grip=-1.0 grasp=False success=False
case=00 t=080 phase=LIFT->LIFT eef=[0.00386, 0.02305, 1.15006] obj=[0.01095, 0.02276, 0.81989] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=090 phase=LIFT->LIFT eef=[0.00386, 0.02298, 1.15006] obj=[0.01095, 0.02276, 0.81989] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=100 phase=LIFT->LIFT eef=[0.00386, 0.02292, 1.15005] obj=[0.01095, 0.02276, 0.81989] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=110 phase=LIFT->LIFT eef=[0.00386, 0.02288, 1.15004] obj=[0.01095, 0.02276, 0.81989] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=120 phase=LIFT->LIFT eef=[0.00386, 0.02284, 1.15004] obj=[0.01095, 0.02276, 0.81989] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=130 phase=LIFT->LIFT eef=[0.00386, 0.0228, 1.15003] obj=[0.01095, 0.02276, 0.81989] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=140 phase=LIFT->LIFT eef=[0.00386, 0.02278, 1.15003] obj=[0.01095, 0.02276, 0.81989] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=150 phase=LIFT->LIFT eef=[0.00386, 0.02276, 1.15003] obj=[0.01095, 0.02276, 0.81989] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=160 phase=LIFT->LIFT eef=[0.00386, 0.02274, 1.15002] obj=[0.01095, 0.02276, 0.81989] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=170 phase=LIFT->LIFT eef=[0.00386, 0.02272, 1.15002] obj=[0.01095, 0.02276, 0.81989] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=180 phase=LIFT->LIFT eef=[0.00386, 0.02271, 1.15002] obj=[0.01095, 0.02276, 0.81989] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=190 phase=LIFT->LIFT eef=[0.00386, 0.02271, 1.15002] obj=[0.01095, 0.02276, 0.81989] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=200 phase=LIFT->LIFT eef=[0.00386, 0.0227, 1.15001] obj=[0.01095, 0.02276, 0.81989] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=210 phase=LIFT->LIFT eef=[0.00386, 0.02269, 1.15001] obj=[0.01095, 0.02276, 0.81989] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=220 phase=LIFT->LIFT eef=[0.00386, 0.02269, 1.15001] obj=[0.01095, 0.02276, 0.81989] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=230 phase=LIFT->LIFT eef=[0.00386, 0.02269, 1.15001] obj=[0.01095, 0.02276, 0.81989] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=240 phase=LIFT->LIFT eef=[0.00386, 0.02269, 1.15001] obj=[0.01095, 0.02276, 0.81989] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=250 phase=LIFT->LIFT eef=[0.00386, 0.02269, 1.15001] obj=[0.01095, 0.02276, 0.81989] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=260 phase=LIFT->LIFT eef=[0.00386, 0.02269, 1.15001] obj=[0.01095, 0.02276, 0.81989] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=270 phase=LIFT->LIFT eef=[0.00386, 0.02269, 1.15001] obj=[0.01095, 0.02276, 0.81989] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=280 phase=LIFT->LIFT eef=[0.00386, 0.02268, 1.15001] obj=[0.01095, 0.02276, 0.81989] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=290 phase=LIFT->LIFT eef=[0.00386, 0.02268, 1.15] obj=[0.01095, 0.02276, 0.81989] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=300 phase=LIFT->LIFT eef=[0.00386, 0.02268, 1.15] obj=[0.01095, 0.02276, 0.81989] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=310 phase=LIFT->LIFT eef=[0.00386, 0.02268, 1.15] obj=[0.01095, 0.02276, 0.81989] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=320 phase=LIFT->LIFT eef=[0.00386, 0.02268, 1.15] obj=[0.01095, 0.02276, 0.81989] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=330 phase=LIFT->LIFT eef=[0.00386, 0.02268, 1.15] obj=[0.01095, 0.02276, 0.81989] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=340 phase=LIFT->LIFT eef=[0.00386, 0.02268, 1.15] obj=[0.01095, 0.02276, 0.81989] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=350 phase=LIFT->LIFT eef=[0.00386, 0.02268, 1.15] obj=[0.01095, 0.02276, 0.81989] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=360 phase=LIFT->LIFT eef=[0.00386, 0.02268, 1.15] obj=[0.01095, 0.02276, 0.81989] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=370 phase=LIFT->LIFT eef=[0.00386, 0.02268, 1.15] obj=[0.01095, 0.02276, 0.81989] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=380 phase=LIFT->LIFT eef=[0.00386, 0.02268, 1.15] obj=[0.01095, 0.02276, 0.81989] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=390 phase=LIFT->LIFT eef=[0.00386, 0.02267, 1.15] obj=[0.01095, 0.02276, 0.81989] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=400 phase=LIFT->LIFT eef=[0.00386, 0.02267, 1.15] obj=[0.01095, 0.02276, 0.81989] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=410 phase=LIFT->LIFT eef=[0.00386, 0.02267, 1.15] obj=[0.01095, 0.02276, 0.81989] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=420 phase=LIFT->LIFT eef=[0.00386, 0.02267, 1.15] obj=[0.01095, 0.02276, 0.81989] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=430 phase=LIFT->LIFT eef=[0.00386, 0.02267, 1.15] obj=[0.01095, 0.02276, 0.81989] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=440 phase=LIFT->LIFT eef=[0.00386, 0.02267, 1.15] obj=[0.01095, 0.02276, 0.81989] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=450 phase=LIFT->LIFT eef=[0.00386, 0.02267, 1.15] obj=[0.01095, 0.02276, 0.81989] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=460 phase=LIFT->LIFT eef=[0.00386, 0.02267, 1.15] obj=[0.01095, 0.02276, 0.81989] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=470 phase=LIFT->LIFT eef=[0.00386, 0.02267, 1.15] obj=[0.01095, 0.02276, 0.81989] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=480 phase=LIFT->LIFT eef=[0.00386, 0.02267, 1.15] obj=[0.01095, 0.02276, 0.81989] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=490 phase=LIFT->LIFT eef=[0.00386, 0.02267, 1.15] obj=[0.01095, 0.02276, 0.81989] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=01 t=000 phase=APPROACH->APPROACH eef=[-0.101, 0.00827, 1.01149] obj=[-0.01784, 0.01026, 0.81969] a_xyz=[1.0, 0.008, 1.0] grip=1.0 grasp=False success=False
case=01 t=010 phase=DESCEND->DESCEND eef=[-0.01631, 0.01019, 1.06424] obj=[-0.01784, 0.01026, 0.81995] a_xyz=[0.065, 0.005, -1.0] grip=1.0 grasp=False success=False
case=01 t=020 phase=GRASP->GRASP eef=[-0.0179, 0.00985, 1.00082] obj=[-0.01784, 0.01026, 0.81995] a_xyz=[-0.012, 0.008, -0.039] grip=-1.0 grasp=False success=False
case=01 t=030 phase=GRASP->GRASP eef=[-0.01821, 0.00999, 0.99976] obj=[-0.01784, 0.01026, 0.81995] a_xyz=[0.009, 0.006, 0.005] grip=-1.0 grasp=False success=False
case=01 t=040 phase=LIFT->LIFT eef=[-0.01889, 0.01026, 1.01491] obj=[-0.01784, 0.01026, 0.81995] a_xyz=[0.0, 0.0, 1.0] grip=-1.0 grasp=False success=False
case=01 t=050 phase=LIFT->LIFT eef=[-0.02744, 0.01113, 1.13421] obj=[-0.01784, 0.01026, 0.81995] a_xyz=[0.0, 0.0, 0.48] grip=-1.0 grasp=False success=False
case=01 t=060 phase=LIFT->LIFT eef=[-0.02607, 0.01102, 1.15034] obj=[-0.01784, 0.01026, 0.81995] a_xyz=[0.0, 0.0, -0.008] grip=-1.0 grasp=False success=False
case=01 t=070 phase=LIFT->LIFT eef=[-0.02609, 0.01087, 1.15007] obj=[-0.01784, 0.01026, 0.81995] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=01 t=080 phase=LIFT->LIFT eef=[-0.02609, 0.01075, 1.15006] obj=[-0.01784, 0.01026, 0.81995] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=01 t=090 phase=LIFT->LIFT eef=[-0.02609, 0.01064, 1.15005] obj=[-0.01784, 0.01026, 0.81995] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=01 t=100 phase=LIFT->LIFT eef=[-0.02609, 0.01055, 1.15005] obj=[-0.01784, 0.01026, 0.81995] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=01 t=110 phase=LIFT->LIFT eef=[-0.0261, 0.01048, 1.15004] obj=[-0.01784, 0.01026, 0.81995] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=01 t=120 phase=LIFT->LIFT eef=[-0.0261, 0.01042, 1.15004] obj=[-0.01784, 0.01026, 0.81995] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=01 t=130 phase=LIFT->LIFT eef=[-0.0261, 0.01037, 1.15003] obj=[-0.01784, 0.01026, 0.81995] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=01 t=140 phase=LIFT->LIFT eef=[-0.0261, 0.01033, 1.15003] obj=[-0.01784, 0.01026, 0.81995] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=01 t=150 phase=LIFT->LIFT eef=[-0.0261, 0.0103, 1.15003] obj=[-0.01784, 0.01026, 0.81995] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=01 t=160 phase=LIFT->LIFT eef=[-0.0261, 0.01027, 1.15002] obj=[-0.01784, 0.01026, 0.81995] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=01 t=170 phase=LIFT->LIFT eef=[-0.0261, 0.01025, 1.15002] obj=[-0.01784, 0.01026, 0.81995] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=01 t=180 phase=LIFT->LIFT eef=[-0.0261, 0.01023, 1.15002] obj=[-0.01784, 0.01026, 0.81995] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=01 t=190 phase=LIFT->LIFT eef=[-0.0261, 0.01022, 1.15002] obj=[-0.01784, 0.01026, 0.81995] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=01 t=200 phase=LIFT->LIFT eef=[-0.0261, 0.0102, 1.15001] obj=[-0.01784, 0.01026, 0.81995] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=01 t=210 phase=LIFT->LIFT eef=[-0.0261, 0.0102, 1.15001] obj=[-0.01784, 0.01026, 0.81995] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=01 t=220 phase=LIFT->LIFT eef=[-0.0261, 0.01019, 1.15001] obj=[-0.01784, 0.01026, 0.81995] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=01 t=230 phase=LIFT->LIFT eef=[-0.0261, 0.01018, 1.15001] obj=[-0.01784, 0.01026, 0.81995] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=01 t=240 phase=LIFT->LIFT eef=[-0.0261, 0.01018, 1.15001] obj=[-0.01784, 0.01026, 0.81995] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=01 t=250 phase=LIFT->LIFT eef=[-0.0261, 0.01018, 1.15001] obj=[-0.01784, 0.01026, 0.81995] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=01 t=260 phase=LIFT->LIFT eef=[-0.0261, 0.01018, 1.15001] obj=[-0.01784, 0.01026, 0.81995] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=01 t=270 phase=LIFT->LIFT eef=[-0.0261, 0.01018, 1.15001] obj=[-0.01784, 0.01026, 0.81995] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=01 t=280 phase=LIFT->LIFT eef=[-0.0261, 0.01017, 1.15001] obj=[-0.01784, 0.01026, 0.81995] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=01 t=290 phase=LIFT->LIFT eef=[-0.0261, 0.01017, 1.15] obj=[-0.01784, 0.01026, 0.81995] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=01 t=300 phase=LIFT->LIFT eef=[-0.0261, 0.01017, 1.15] obj=[-0.01784, 0.01026, 0.81995] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=01 t=310 phase=LIFT->LIFT eef=[-0.0261, 0.01017, 1.15] obj=[-0.01784, 0.01026, 0.81995] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=01 t=320 phase=LIFT->LIFT eef=[-0.0261, 0.01017, 1.15] obj=[-0.01784, 0.01026, 0.81995] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=01 t=330 phase=LIFT->LIFT eef=[-0.0261, 0.01017, 1.15] obj=[-0.01784, 0.01026, 0.81995] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=01 t=340 phase=LIFT->LIFT eef=[-0.0261, 0.01017, 1.15] obj=[-0.01784, 0.01026, 0.81995] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=01 t=350 phase=LIFT->LIFT eef=[-0.0261, 0.01017, 1.15] obj=[-0.01784, 0.01026, 0.81995] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=01 t=360 phase=LIFT->LIFT eef=[-0.0261, 0.01017, 1.15] obj=[-0.01784, 0.01026, 0.81995] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=01 t=370 phase=LIFT->LIFT eef=[-0.0261, 0.01017, 1.15] obj=[-0.01784, 0.01026, 0.81995] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=01 t=380 phase=LIFT->LIFT eef=[-0.0261, 0.01017, 1.15] obj=[-0.01784, 0.01026, 0.81995] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=01 t=390 phase=LIFT->LIFT eef=[-0.0261, 0.01016, 1.15] obj=[-0.01784, 0.01026, 0.81995] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=01 t=400 phase=LIFT->LIFT eef=[-0.0261, 0.01016, 1.15] obj=[-0.01784, 0.01026, 0.81995] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=01 t=410 phase=LIFT->LIFT eef=[-0.0261, 0.01016, 1.15] obj=[-0.01784, 0.01026, 0.81995] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=01 t=420 phase=LIFT->LIFT eef=[-0.02611, 0.01016, 1.15] obj=[-0.01784, 0.01026, 0.81995] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=01 t=430 phase=LIFT->LIFT eef=[-0.02611, 0.01016, 1.15] obj=[-0.01784, 0.01026, 0.81995] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=01 t=440 phase=LIFT->LIFT eef=[-0.02611, 0.01016, 1.15] obj=[-0.01784, 0.01026, 0.81995] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=01 t=450 phase=LIFT->LIFT eef=[-0.02611, 0.01016, 1.15] obj=[-0.01784, 0.01026, 0.81995] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=01 t=460 phase=LIFT->LIFT eef=[-0.02611, 0.01016, 1.15] obj=[-0.01784, 0.01026, 0.81995] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=01 t=470 phase=LIFT->LIFT eef=[-0.02611, 0.01016, 1.15] obj=[-0.01784, 0.01026, 0.81995] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=01 t=480 phase=LIFT->LIFT eef=[-0.02611, 0.01016, 1.15] obj=[-0.01784, 0.01026, 0.81995] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=01 t=490 phase=LIFT->LIFT eef=[-0.02611, 0.01016, 1.15] obj=[-0.01784, 0.01026, 0.81995] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
Complete diagnostic trace written to /content/PhaseForge/outputs/_gates/debug_lift_2026-08-15_16-35-34.json


[robosuite WARNING] No private macro file found! (macros.py:53)
[robosuite WARNING] It is recommended to use a private macro file (macros.py:54)
[robosuite WARNING] To setup, run: python /content/PhaseForge/.venv/lib/python3.12/site-packages/robosuite/scripts/setup_macros.py (macros.py:55)
[robosuite WARNING] Could not import robosuite_models. Some robots may not be available. If you want to use these robots, please install robosuite_models from source (https://github.com/ARISE-Initiative/robosuite_models) or through pip install. (__init__.py:30)
[robosuite WARNING] Could not load the mink-based whole-body IK. Make sure you install related import properly, otherwise you will not be able to use the default IK controller setting for GR1 robot. (__init__.py:40)
Task: Lift; bank: a7d3953c0afcf560; cases: [0]
Controller metadata: {
  "robot0.right": {
    "__class__": "OperationalSpaceController",
    "input_type": "delta",
    "input_ref_frame": "world",
    "control_delta": null,
    "input_min": [
      -1.0,
      -1.0,
      -1.0,
      -1.0,
      -1.0,
      -1.0
    ],
    "input_max": [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    "output_min": [
      -0.05,
      -0.05,
      -0.05,
      -0.5,
      -0.5,
      -0.5
    ],
    "output_max": [
      0.05,
      0.05,
      0.05,
      0.5,
      0.5,
      0.5
    ]
  },
  "robot0.right_gripper": {
    "__class__": "SimpleGripController",
    "input_type": null,
    "input_ref_frame": null,
    "control_delta": null,
    "input_min": [
      -1.0,
      -1.0
    ],
    "input_max": [
      1.0,
      1.0
    ],
    "output_min": [
      -1.0,
      -1.0
    ],
    "output_max": [
      1.0,
      1.0
    ]
  }
}
case=00 t=000 phase=APPROACH->APPROACH eef=[-0.10944, -0.0071, 1.01196] obj=[0.01095, 0.02276, 0.82078] a_xyz=[1.0, 0.696, 1.0] grip=1.0 grasp=False success=False
case=00 t=001 phase=APPROACH->APPROACH eef=[-0.10095, -0.0001, 1.02226] obj=[0.01095, 0.02276, 0.81948] a_xyz=[1.0, 0.597, 1.0] grip=1.0 grasp=False success=False
case=00 t=002 phase=APPROACH->APPROACH eef=[-0.09182, 0.00582, 1.03296] obj=[0.01095, 0.02276, 0.82058] a_xyz=[1.0, 0.457, 1.0] grip=1.0 grasp=False success=False
case=00 t=003 phase=APPROACH->APPROACH eef=[-0.08179, 0.01108, 1.04428] obj=[0.01095, 0.02276, 0.82109] a_xyz=[1.0, 0.339, 1.0] grip=1.0 grasp=False success=False
case=00 t=004 phase=APPROACH->APPROACH eef=[-0.07113, 0.01553, 1.05607] obj=[0.01095, 0.02276, 0.82132] a_xyz=[1.0, 0.234, 1.0] grip=1.0 grasp=False success=False
case=00 t=005 phase=APPROACH->APPROACH eef=[-0.05995, 0.019, 1.06769] obj=[0.01095, 0.02276, 0.82144] a_xyz=[1.0, 0.145, 0.905] grip=1.0 grasp=False success=False
case=00 t=006 phase=APPROACH->APPROACH eef=[-0.04822, 0.02146, 1.07778] obj=[0.01095, 0.02276, 0.82149] a_xyz=[1.0, 0.075, 0.675] grip=1.0 grasp=False success=False
case=00 t=007 phase=APPROACH->APPROACH eef=[-0.03593, 0.02298, 1.08558] obj=[0.01095, 0.02276, 0.82152] a_xyz=[1.0, 0.026, 0.474] grip=1.0 grasp=False success=False
case=00 t=008 phase=APPROACH->APPROACH eef=[-0.02352, 0.02375, 1.09124] obj=[0.01095, 0.02276, 0.82153] a_xyz=[0.938, -0.004, 0.319] grip=1.0 grasp=False success=False
case=00 t=009 phase=APPROACH->APPROACH eef=[-0.0124, 0.02405, 1.09537] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.69, -0.02, 0.206] grip=1.0 grasp=False success=False
case=00 t=010 phase=APPROACH->DESCEND eef=[-0.00307, 0.02404, 1.09241] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.467, -0.026, -1.0] grip=1.0 grasp=False success=False
case=00 t=011 phase=DESCEND->DESCEND eef=[0.00475, 0.0238, 1.08294] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.28, -0.026, -1.0] grip=1.0 grasp=False success=False
case=00 t=012 phase=DESCEND->DESCEND eef=[0.01044, 0.02355, 1.07213] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.124, -0.021, -1.0] grip=1.0 grasp=False success=False
case=00 t=013 phase=DESCEND->DESCEND eef=[0.01414, 0.02333, 1.06093] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.01, -0.016, -1.0] grip=1.0 grasp=False success=False
case=00 t=014 phase=DESCEND->DESCEND eef=[0.01624, 0.02316, 1.04948] obj=[0.01095, 0.02276, 0.82154] a_xyz=[-0.064, -0.011, -1.0] grip=1.0 grasp=False success=False
case=00 t=015 phase=DESCEND->GRASP eef=[0.01714, 0.02302, 1.03799] obj=[0.01095, 0.02276, 0.82154] a_xyz=[-0.106, -0.008, -0.959] grip=-1.0 grasp=False success=False
case=00 t=016 phase=GRASP->GRASP eef=[0.01713, 0.02291, 1.02774] obj=[0.01095, 0.02276, 0.82154] a_xyz=[-0.124, -0.005, -0.729] grip=-1.0 grasp=False success=False
case=00 t=017 phase=GRASP->GRASP eef=[0.01648, 0.02285, 1.0197] obj=[0.01095, 0.02276, 0.82154] a_xyz=[-0.124, -0.003, -0.524] grip=-1.0 grasp=False success=False
case=00 t=018 phase=GRASP->GRASP eef=[0.01548, 0.02283, 1.01379] obj=[0.01095, 0.02276, 0.82154] a_xyz=[-0.11, -0.002, -0.363] grip=-1.0 grasp=False success=False
case=00 t=019 phase=GRASP->GRASP eef=[0.01437, 0.02282, 1.00957] obj=[0.01095, 0.02276, 0.82154] a_xyz=[-0.091, -0.001, -0.245] grip=-1.0 grasp=False success=False
case=00 t=020 phase=GRASP->GRASP eef=[0.0133, 0.0228, 1.00662] obj=[0.01095, 0.02276, 0.82154] a_xyz=[-0.068, -0.001, -0.16] grip=-1.0 grasp=False success=False
case=00 t=021 phase=GRASP->GRASP eef=[0.01236, 0.02279, 1.0046] obj=[0.01095, 0.02276, 0.82154] a_xyz=[-0.047, -0.001, -0.102] grip=-1.0 grasp=False success=False
case=00 t=022 phase=GRASP->GRASP eef=[0.01161, 0.02277, 1.00323] obj=[0.01095, 0.02276, 0.82154] a_xyz=[-0.028, -0.001, -0.061] grip=-1.0 grasp=False success=False
case=00 t=023 phase=GRASP->GRASP eef=[0.01105, 0.02274, 1.00234] obj=[0.01095, 0.02276, 0.82154] a_xyz=[-0.013, -0.0, -0.034] grip=-1.0 grasp=False success=False
case=00 t=024 phase=GRASP->GRASP eef=[0.01066, 0.02274, 1.00177] obj=[0.01095, 0.02276, 0.82154] a_xyz=[-0.002, 0.0, -0.016] grip=-1.0 grasp=False success=False
case=00 t=025 phase=GRASP->GRASP eef=[0.01043, 0.02274, 1.00143] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.006, 0.0, -0.004] grip=-1.0 grasp=False success=False
case=00 t=026 phase=GRASP->GRASP eef=[0.01033, 0.02276, 1.00126] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.01, 0.0, 0.002] grip=-1.0 grasp=False success=False
case=00 t=027 phase=GRASP->GRASP eef=[0.01032, 0.02278, 1.0012] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.013, 0.0, 0.006] grip=-1.0 grasp=False success=False
case=00 t=028 phase=GRASP->GRASP eef=[0.01036, 0.0228, 1.00121] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.013, -0.0, 0.007] grip=-1.0 grasp=False success=False
case=00 t=029 phase=GRASP->GRASP eef=[0.01042, 0.02282, 1.00124] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.012, -0.001, 0.007] grip=-1.0 grasp=False success=False
case=00 t=030 phase=GRASP->GRASP eef=[0.01049, 0.02283, 1.00129] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.011, -0.001, 0.006] grip=-1.0 grasp=False success=False
case=00 t=031 phase=GRASP->GRASP eef=[0.01055, 0.02284, 1.00134] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.009, -0.001, 0.005] grip=-1.0 grasp=False success=False
case=00 t=032 phase=GRASP->GRASP eef=[0.01061, 0.02285, 1.00138] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.008, -0.002, 0.004] grip=-1.0 grasp=False success=False
case=00 t=033 phase=GRASP->GRASP eef=[0.01065, 0.02286, 1.00141] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.007, -0.002, 0.003] grip=-1.0 grasp=False success=False
case=00 t=034 phase=GRASP->GRASP eef=[0.01069, 0.02286, 1.00144] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.006, -0.002, 0.003] grip=-1.0 grasp=False success=False
case=00 t=035 phase=GRASP->GRASP eef=[0.01071, 0.02286, 1.00145] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.005, -0.002, 0.002] grip=-1.0 grasp=False success=False
case=00 t=036 phase=GRASP->GRASP eef=[0.01073, 0.02286, 1.00147] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.005, -0.002, 0.002] grip=-1.0 grasp=False success=False
case=00 t=037 phase=GRASP->GRASP eef=[0.01075, 0.02286, 1.00148] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.004, -0.002, 0.002] grip=-1.0 grasp=False success=False
case=00 t=038 phase=GRASP->GRASP eef=[0.01076, 0.02286, 1.00149] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.004, -0.002, 0.001] grip=-1.0 grasp=False success=False
case=00 t=039 phase=GRASP->GRASP eef=[0.01077, 0.02286, 1.0015] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.004, -0.002, 0.001] grip=-1.0 grasp=False success=False
case=00 t=040 phase=GRASP->LIFT eef=[0.01078, 0.02285, 1.0015] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.004, -0.002, 0.001] grip=-1.0 grasp=False success=False
case=00 t=041 phase=LIFT->LIFT eef=[0.01081, 0.02292, 1.00595] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, 1.0] grip=-1.0 grasp=False success=False
case=00 t=042 phase=LIFT->LIFT eef=[0.01023, 0.02311, 1.01673] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, 1.0] grip=-1.0 grasp=False success=False
case=00 t=043 phase=LIFT->LIFT eef=[0.00936, 0.02332, 1.029] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, 1.0] grip=-1.0 grasp=False success=False
case=00 t=044 phase=LIFT->LIFT eef=[0.00848, 0.02349, 1.0416] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, 1.0] grip=-1.0 grasp=False success=False
case=00 t=045 phase=LIFT->LIFT eef=[0.00762, 0.02362, 1.05429] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, 1.0] grip=-1.0 grasp=False success=False
case=00 t=046 phase=LIFT->LIFT eef=[0.0068, 0.02371, 1.06702] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, 1.0] grip=-1.0 grasp=False success=False
case=00 t=047 phase=LIFT->LIFT eef=[0.00601, 0.02378, 1.07979] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, 1.0] grip=-1.0 grasp=False success=False
case=00 t=048 phase=LIFT->LIFT eef=[0.00524, 0.02382, 1.09256] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, 1.0] grip=-1.0 grasp=False success=False
case=00 t=049 phase=LIFT->LIFT eef=[0.00448, 0.02385, 1.10534] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, 1.0] grip=-1.0 grasp=False success=False
case=00 t=050 phase=LIFT->LIFT eef=[0.00373, 0.02386, 1.11752] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, 0.893] grip=-1.0 grasp=False success=False
case=00 t=051 phase=LIFT->LIFT eef=[0.00315, 0.02384, 1.12777] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, 0.65] grip=-1.0 grasp=False success=False
case=00 t=052 phase=LIFT->LIFT eef=[0.00283, 0.02381, 1.13544] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, 0.445] grip=-1.0 grasp=False success=False
case=00 t=053 phase=LIFT->LIFT eef=[0.00276, 0.02377, 1.14085] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, 0.291] grip=-1.0 grasp=False success=False
case=00 t=054 phase=LIFT->LIFT eef=[0.00285, 0.02374, 1.14452] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, 0.183] grip=-1.0 grasp=False success=False
case=00 t=055 phase=LIFT->LIFT eef=[0.00302, 0.02372, 1.14691] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, 0.11] grip=-1.0 grasp=False success=False
case=00 t=056 phase=LIFT->LIFT eef=[0.00322, 0.02372, 1.14843] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, 0.062] grip=-1.0 grasp=False success=False
case=00 t=057 phase=LIFT->LIFT eef=[0.00342, 0.02372, 1.14934] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, 0.031] grip=-1.0 grasp=False success=False
case=00 t=058 phase=LIFT->LIFT eef=[0.00359, 0.02373, 1.14986] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, 0.013] grip=-1.0 grasp=False success=False
case=00 t=059 phase=LIFT->LIFT eef=[0.00373, 0.02372, 1.15014] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, 0.003] grip=-1.0 grasp=False success=False
case=00 t=060 phase=LIFT->LIFT eef=[0.00383, 0.02372, 1.15026] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.003] grip=-1.0 grasp=False success=False
case=00 t=061 phase=LIFT->LIFT eef=[0.00389, 0.02371, 1.15028] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.005] grip=-1.0 grasp=False success=False
case=00 t=062 phase=LIFT->LIFT eef=[0.00392, 0.02369, 1.15025] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.006] grip=-1.0 grasp=False success=False
case=00 t=063 phase=LIFT->LIFT eef=[0.00391, 0.02368, 1.1502] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.005] grip=-1.0 grasp=False success=False
case=00 t=064 phase=LIFT->LIFT eef=[0.00391, 0.02366, 1.15016] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.004] grip=-1.0 grasp=False success=False
case=00 t=065 phase=LIFT->LIFT eef=[0.0039, 0.02365, 1.15013] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.003] grip=-1.0 grasp=False success=False
case=00 t=066 phase=LIFT->LIFT eef=[0.0039, 0.02363, 1.15011] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.003] grip=-1.0 grasp=False success=False
case=00 t=067 phase=LIFT->LIFT eef=[0.0039, 0.02362, 1.15009] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.002] grip=-1.0 grasp=False success=False
case=00 t=068 phase=LIFT->LIFT eef=[0.0039, 0.0236, 1.15009] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.002] grip=-1.0 grasp=False success=False
case=00 t=069 phase=LIFT->LIFT eef=[0.0039, 0.02359, 1.15008] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.002] grip=-1.0 grasp=False success=False
case=00 t=070 phase=LIFT->LIFT eef=[0.0039, 0.02358, 1.15008] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.002] grip=-1.0 grasp=False success=False
case=00 t=071 phase=LIFT->LIFT eef=[0.0039, 0.02356, 1.15007] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.002] grip=-1.0 grasp=False success=False
case=00 t=072 phase=LIFT->LIFT eef=[0.0039, 0.02355, 1.15007] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=073 phase=LIFT->LIFT eef=[0.0039, 0.02354, 1.15007] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=074 phase=LIFT->LIFT eef=[0.0039, 0.02353, 1.15007] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=075 phase=LIFT->LIFT eef=[0.0039, 0.02351, 1.15007] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=076 phase=LIFT->LIFT eef=[0.0039, 0.0235, 1.15007] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=077 phase=LIFT->LIFT eef=[0.0039, 0.02349, 1.15007] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=078 phase=LIFT->LIFT eef=[0.0039, 0.02348, 1.15007] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=079 phase=LIFT->LIFT eef=[0.0039, 0.02347, 1.15006] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=080 phase=LIFT->LIFT eef=[0.0039, 0.02346, 1.15006] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=081 phase=LIFT->LIFT eef=[0.0039, 0.02345, 1.15006] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=082 phase=LIFT->LIFT eef=[0.0039, 0.02344, 1.15006] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=083 phase=LIFT->LIFT eef=[0.0039, 0.02343, 1.15006] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=084 phase=LIFT->LIFT eef=[0.0039, 0.02342, 1.15006] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=085 phase=LIFT->LIFT eef=[0.0039, 0.02341, 1.15006] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=086 phase=LIFT->LIFT eef=[0.0039, 0.0234, 1.15006] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=087 phase=LIFT->LIFT eef=[0.0039, 0.02339, 1.15006] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=088 phase=LIFT->LIFT eef=[0.0039, 0.02338, 1.15006] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=089 phase=LIFT->LIFT eef=[0.0039, 0.02337, 1.15006] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=090 phase=LIFT->LIFT eef=[0.0039, 0.02336, 1.15006] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=091 phase=LIFT->LIFT eef=[0.0039, 0.02335, 1.15006] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=092 phase=LIFT->LIFT eef=[0.0039, 0.02334, 1.15005] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=093 phase=LIFT->LIFT eef=[0.0039, 0.02333, 1.15005] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=094 phase=LIFT->LIFT eef=[0.0039, 0.02333, 1.15005] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=095 phase=LIFT->LIFT eef=[0.0039, 0.02332, 1.15005] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=096 phase=LIFT->LIFT eef=[0.0039, 0.02331, 1.15005] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=097 phase=LIFT->LIFT eef=[0.0039, 0.0233, 1.15005] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=098 phase=LIFT->LIFT eef=[0.0039, 0.02329, 1.15005] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=099 phase=LIFT->LIFT eef=[0.0039, 0.02329, 1.15005] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=100 phase=LIFT->LIFT eef=[0.0039, 0.02328, 1.15005] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=101 phase=LIFT->LIFT eef=[0.0039, 0.02327, 1.15005] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=102 phase=LIFT->LIFT eef=[0.0039, 0.02326, 1.15005] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=103 phase=LIFT->LIFT eef=[0.0039, 0.02326, 1.15005] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=104 phase=LIFT->LIFT eef=[0.0039, 0.02325, 1.15005] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=105 phase=LIFT->LIFT eef=[0.0039, 0.02324, 1.15005] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=106 phase=LIFT->LIFT eef=[0.0039, 0.02324, 1.15005] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=107 phase=LIFT->LIFT eef=[0.0039, 0.02323, 1.15005] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=108 phase=LIFT->LIFT eef=[0.0039, 0.02322, 1.15004] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=109 phase=LIFT->LIFT eef=[0.0039, 0.02322, 1.15004] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=110 phase=LIFT->LIFT eef=[0.0039, 0.02321, 1.15004] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=111 phase=LIFT->LIFT eef=[0.0039, 0.02321, 1.15004] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=112 phase=LIFT->LIFT eef=[0.0039, 0.0232, 1.15004] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=113 phase=LIFT->LIFT eef=[0.0039, 0.02319, 1.15004] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=114 phase=LIFT->LIFT eef=[0.0039, 0.02319, 1.15004] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=115 phase=LIFT->LIFT eef=[0.0039, 0.02318, 1.15004] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=116 phase=LIFT->LIFT eef=[0.0039, 0.02318, 1.15004] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=117 phase=LIFT->LIFT eef=[0.0039, 0.02317, 1.15004] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=118 phase=LIFT->LIFT eef=[0.0039, 0.02317, 1.15004] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=119 phase=LIFT->LIFT eef=[0.0039, 0.02316, 1.15004] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=120 phase=LIFT->LIFT eef=[0.0039, 0.02316, 1.15004] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=121 phase=LIFT->LIFT eef=[0.0039, 0.02315, 1.15004] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=122 phase=LIFT->LIFT eef=[0.0039, 0.02315, 1.15004] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=123 phase=LIFT->LIFT eef=[0.0039, 0.02314, 1.15004] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=124 phase=LIFT->LIFT eef=[0.0039, 0.02314, 1.15004] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=125 phase=LIFT->LIFT eef=[0.0039, 0.02313, 1.15004] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=126 phase=LIFT->LIFT eef=[0.0039, 0.02313, 1.15004] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=127 phase=LIFT->LIFT eef=[0.0039, 0.02312, 1.15004] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=128 phase=LIFT->LIFT eef=[0.0039, 0.02312, 1.15003] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=129 phase=LIFT->LIFT eef=[0.0039, 0.02311, 1.15003] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=130 phase=LIFT->LIFT eef=[0.0039, 0.02311, 1.15003] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=131 phase=LIFT->LIFT eef=[0.0039, 0.0231, 1.15003] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=132 phase=LIFT->LIFT eef=[0.0039, 0.0231, 1.15003] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=133 phase=LIFT->LIFT eef=[0.0039, 0.0231, 1.15003] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=134 phase=LIFT->LIFT eef=[0.0039, 0.02309, 1.15003] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=135 phase=LIFT->LIFT eef=[0.0039, 0.02309, 1.15003] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=136 phase=LIFT->LIFT eef=[0.0039, 0.02309, 1.15003] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=137 phase=LIFT->LIFT eef=[0.0039, 0.02308, 1.15003] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=138 phase=LIFT->LIFT eef=[0.0039, 0.02308, 1.15003] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=139 phase=LIFT->LIFT eef=[0.0039, 0.02307, 1.15003] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=140 phase=LIFT->LIFT eef=[0.0039, 0.02307, 1.15003] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=141 phase=LIFT->LIFT eef=[0.0039, 0.02307, 1.15003] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=142 phase=LIFT->LIFT eef=[0.0039, 0.02306, 1.15003] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=143 phase=LIFT->LIFT eef=[0.0039, 0.02306, 1.15003] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=144 phase=LIFT->LIFT eef=[0.0039, 0.02306, 1.15003] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=145 phase=LIFT->LIFT eef=[0.0039, 0.02305, 1.15003] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=146 phase=LIFT->LIFT eef=[0.0039, 0.02305, 1.15003] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=147 phase=LIFT->LIFT eef=[0.0039, 0.02305, 1.15003] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=148 phase=LIFT->LIFT eef=[0.0039, 0.02305, 1.15003] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=149 phase=LIFT->LIFT eef=[0.0039, 0.02304, 1.15003] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=150 phase=LIFT->LIFT eef=[0.0039, 0.02304, 1.15003] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=151 phase=LIFT->LIFT eef=[0.0039, 0.02304, 1.15003] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=152 phase=LIFT->LIFT eef=[0.0039, 0.02303, 1.15003] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=153 phase=LIFT->LIFT eef=[0.0039, 0.02303, 1.15003] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=154 phase=LIFT->LIFT eef=[0.0039, 0.02303, 1.15003] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=155 phase=LIFT->LIFT eef=[0.0039, 0.02303, 1.15002] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.001] grip=-1.0 grasp=False success=False
case=00 t=156 phase=LIFT->LIFT eef=[0.0039, 0.02302, 1.15002] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=157 phase=LIFT->LIFT eef=[0.0039, 0.02302, 1.15002] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=158 phase=LIFT->LIFT eef=[0.0039, 0.02302, 1.15002] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=159 phase=LIFT->LIFT eef=[0.0039, 0.02302, 1.15002] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=160 phase=LIFT->LIFT eef=[0.0039, 0.02301, 1.15002] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=161 phase=LIFT->LIFT eef=[0.0039, 0.02301, 1.15002] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=162 phase=LIFT->LIFT eef=[0.0039, 0.02301, 1.15002] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=163 phase=LIFT->LIFT eef=[0.0039, 0.02301, 1.15002] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=164 phase=LIFT->LIFT eef=[0.0039, 0.023, 1.15002] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=165 phase=LIFT->LIFT eef=[0.0039, 0.023, 1.15002] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=166 phase=LIFT->LIFT eef=[0.0039, 0.023, 1.15002] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=167 phase=LIFT->LIFT eef=[0.0039, 0.023, 1.15002] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=168 phase=LIFT->LIFT eef=[0.0039, 0.023, 1.15002] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=169 phase=LIFT->LIFT eef=[0.0039, 0.02299, 1.15002] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=170 phase=LIFT->LIFT eef=[0.0039, 0.02299, 1.15002] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=171 phase=LIFT->LIFT eef=[0.0039, 0.02299, 1.15002] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=172 phase=LIFT->LIFT eef=[0.0039, 0.02299, 1.15002] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=173 phase=LIFT->LIFT eef=[0.0039, 0.02299, 1.15002] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=174 phase=LIFT->LIFT eef=[0.0039, 0.02299, 1.15002] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=175 phase=LIFT->LIFT eef=[0.0039, 0.02298, 1.15002] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=176 phase=LIFT->LIFT eef=[0.0039, 0.02298, 1.15002] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=177 phase=LIFT->LIFT eef=[0.0039, 0.02298, 1.15002] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=178 phase=LIFT->LIFT eef=[0.0039, 0.02298, 1.15002] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=179 phase=LIFT->LIFT eef=[0.0039, 0.02298, 1.15002] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=180 phase=LIFT->LIFT eef=[0.0039, 0.02298, 1.15002] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=181 phase=LIFT->LIFT eef=[0.0039, 0.02297, 1.15002] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=182 phase=LIFT->LIFT eef=[0.0039, 0.02297, 1.15002] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=183 phase=LIFT->LIFT eef=[0.0039, 0.02297, 1.15002] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=184 phase=LIFT->LIFT eef=[0.0039, 0.02297, 1.15002] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=185 phase=LIFT->LIFT eef=[0.0039, 0.02297, 1.15002] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=186 phase=LIFT->LIFT eef=[0.0039, 0.02297, 1.15002] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=187 phase=LIFT->LIFT eef=[0.0039, 0.02297, 1.15002] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=188 phase=LIFT->LIFT eef=[0.0039, 0.02297, 1.15002] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=189 phase=LIFT->LIFT eef=[0.0039, 0.02296, 1.15002] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=190 phase=LIFT->LIFT eef=[0.0039, 0.02296, 1.15002] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=191 phase=LIFT->LIFT eef=[0.0039, 0.02296, 1.15002] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=192 phase=LIFT->LIFT eef=[0.0039, 0.02296, 1.15002] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=193 phase=LIFT->LIFT eef=[0.0039, 0.02296, 1.15002] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=194 phase=LIFT->LIFT eef=[0.0039, 0.02296, 1.15002] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=195 phase=LIFT->LIFT eef=[0.0039, 0.02296, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=196 phase=LIFT->LIFT eef=[0.0039, 0.02296, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=197 phase=LIFT->LIFT eef=[0.0039, 0.02296, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=198 phase=LIFT->LIFT eef=[0.0039, 0.02295, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=199 phase=LIFT->LIFT eef=[0.0039, 0.02295, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=200 phase=LIFT->LIFT eef=[0.0039, 0.02295, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=201 phase=LIFT->LIFT eef=[0.0039, 0.02295, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=202 phase=LIFT->LIFT eef=[0.0039, 0.02295, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=203 phase=LIFT->LIFT eef=[0.0039, 0.02295, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=204 phase=LIFT->LIFT eef=[0.0039, 0.02295, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=205 phase=LIFT->LIFT eef=[0.0039, 0.02295, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=206 phase=LIFT->LIFT eef=[0.0039, 0.02295, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=207 phase=LIFT->LIFT eef=[0.0039, 0.02295, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=208 phase=LIFT->LIFT eef=[0.0039, 0.02295, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=209 phase=LIFT->LIFT eef=[0.0039, 0.02294, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=210 phase=LIFT->LIFT eef=[0.0039, 0.02294, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=211 phase=LIFT->LIFT eef=[0.0039, 0.02294, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=212 phase=LIFT->LIFT eef=[0.0039, 0.02294, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=213 phase=LIFT->LIFT eef=[0.0039, 0.02294, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=214 phase=LIFT->LIFT eef=[0.0039, 0.02294, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=215 phase=LIFT->LIFT eef=[0.0039, 0.02294, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=216 phase=LIFT->LIFT eef=[0.0039, 0.02294, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=217 phase=LIFT->LIFT eef=[0.0039, 0.02294, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=218 phase=LIFT->LIFT eef=[0.0039, 0.02294, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=219 phase=LIFT->LIFT eef=[0.0039, 0.02294, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=220 phase=LIFT->LIFT eef=[0.0039, 0.02294, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=221 phase=LIFT->LIFT eef=[0.0039, 0.02294, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=222 phase=LIFT->LIFT eef=[0.0039, 0.02294, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=223 phase=LIFT->LIFT eef=[0.0039, 0.02294, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=224 phase=LIFT->LIFT eef=[0.0039, 0.02294, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=225 phase=LIFT->LIFT eef=[0.0039, 0.02294, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=226 phase=LIFT->LIFT eef=[0.0039, 0.02294, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=227 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=228 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=229 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=230 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=231 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=232 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=233 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=234 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=235 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=236 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=237 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=238 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=239 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=240 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=241 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=242 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=243 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=244 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=245 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=246 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=247 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=248 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=249 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=250 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=251 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=252 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=253 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=254 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=255 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=256 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=257 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=258 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=259 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=260 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=261 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=262 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=263 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=264 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=265 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=266 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=267 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=268 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=269 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=270 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=271 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=272 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=273 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=274 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=275 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=276 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=277 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=278 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=279 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=280 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=281 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15001] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=282 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=283 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=284 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=285 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=286 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=287 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=288 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=289 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=290 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=291 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=292 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=293 phase=LIFT->LIFT eef=[0.0039, 0.02293, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=294 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=295 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=296 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=297 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=298 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=299 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=300 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=301 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=302 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=303 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=304 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=305 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=306 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=307 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=308 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=309 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=310 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=311 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=312 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=313 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=314 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=315 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=316 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=317 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=318 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=319 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=320 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=321 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=322 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=323 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=324 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=325 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=326 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=327 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=328 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=329 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=330 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=331 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=332 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=333 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=334 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=335 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=336 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=337 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=338 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=339 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=340 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=341 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=342 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=343 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=344 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=345 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=346 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=347 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=348 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=349 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=350 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=351 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=352 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=353 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=354 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=355 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=356 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=357 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=358 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=359 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=360 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=361 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=362 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=363 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=364 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=365 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=366 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=367 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=368 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=369 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=370 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=371 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=372 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=373 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=374 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=375 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=376 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=377 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=378 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=379 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=380 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=381 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=382 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=383 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=384 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=385 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=386 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=387 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=388 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=389 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=390 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=391 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=392 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=393 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=394 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=395 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=396 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=397 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=398 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=399 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=400 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=401 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=402 phase=LIFT->LIFT eef=[0.0039, 0.02292, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=403 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=404 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=405 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=406 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=407 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=408 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=409 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=410 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=411 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=412 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=413 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=414 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=415 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=416 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=417 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=418 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=419 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=420 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=421 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=422 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=423 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=424 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=425 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=426 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=427 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=428 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=429 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=430 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=431 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=432 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=433 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=434 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=435 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=436 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=437 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=438 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=439 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=440 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=441 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=442 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=443 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=444 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=445 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=446 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=447 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=448 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=449 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=450 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=451 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=452 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=453 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=454 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=455 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=456 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=457 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=458 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=459 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=460 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=461 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=462 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=463 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=464 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=465 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=466 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=467 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=468 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=469 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=470 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=471 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=472 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=473 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=474 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=475 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=476 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=477 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=478 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=479 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=480 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=481 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=482 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=483 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=484 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=485 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=486 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=487 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=488 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=489 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=490 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=491 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=492 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=493 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=494 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=495 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=496 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=497 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=498 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
case=00 t=499 phase=LIFT->LIFT eef=[0.0039, 0.02291, 1.15] obj=[0.01095, 0.02276, 0.82154] a_xyz=[0.0, 0.0, -0.0] grip=-1.0 grasp=False success=False
Complete diagnostic trace written to /content/PhaseForge/outputs/_gates/debug_lift_2026-08-15_16-34-35.json