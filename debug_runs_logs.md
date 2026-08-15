{
  "task": "Lift",
  "bank_id": "a7d3953c0afcf560",
  "data_root": "/content/data",
  "config_overrides": [
    "data=lift",
    "eval=rollout"
  ],
  "controller_metadata": {
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
  },
  "traces": [],
  "offset_sweep": [
    {
      "descend_z_offset": 0.0,
      "success": false,
      "native_grasp_observed": true,
      "verified_grasp_observed": false,
      "first_grasp_step": 20,
      "first_verified_grasp_step": null,
      "final_phase": "DESCEND",
      "final_cube_body_pos": [
        0.004425349546738064,
        0.018189085168077875,
        0.8214968072704812
      ],
      "final_gripper_qpos_state": [
        0.0007768852519802749,
        -0.00022266965243034065
      ],
      "gripper_qpos_native_error": null
    },
    {
      "descend_z_offset": 0.01,
      "success": false,
      "native_grasp_observed": true,
      "verified_grasp_observed": false,
      "first_grasp_step": 21,
      "first_verified_grasp_step": null,
      "final_phase": "DESCEND",
      "final_cube_body_pos": [
        0.01926337522203256,
        0.02343198739863362,
        0.8152777546318505
      ],
      "final_gripper_qpos_state": [
        0.0006325131980702281,
        -0.00036705483216792345
      ],
      "gripper_qpos_native_error": null
    },
    {
      "descend_z_offset": 0.02,
      "success": false,
      "native_grasp_observed": false,
      "verified_grasp_observed": false,
      "first_grasp_step": null,
      "first_verified_grasp_step": null,
      "final_phase": "GRASP",
      "final_cube_body_pos": [
        0.010910261552964238,
        0.022723255267213326,
        0.8213409342284295
      ],
      "final_gripper_qpos_state": [
        0.039994437247514725,
        -0.040000177919864655
      ],
      "gripper_qpos_native_error": null
    },
    {
      "descend_z_offset": 0.03,
      "success": false,
      "native_grasp_observed": false,
      "verified_grasp_observed": false,
      "first_grasp_step": null,
      "first_verified_grasp_step": null,
      "final_phase": "GRASP",
      "final_cube_body_pos": [
        0.010954649187624454,
        0.02276298962533474,
        0.8206464650154666
      ],
      "final_gripper_qpos_state": [
        0.03999850153923035,
        -0.03999968618154526
      ],
      "gripper_qpos_native_error": null
    },
    {
      "descend_z_offset": 0.04,
      "success": false,
      "native_grasp_observed": false,
      "verified_grasp_observed": false,
      "first_grasp_step": null,
      "first_verified_grasp_step": null,
      "final_phase": "GRASP",
      "final_cube_body_pos": [
        0.010954649187624454,
        0.02276298962533474,
        0.8214792782481264
      ],
      "final_gripper_qpos_state": [
        0.03999779745936394,
        -0.04000005125999451
      ],
      "gripper_qpos_native_error": null
    },
    {
      "descend_z_offset": 0.05,
      "success": false,
      "native_grasp_observed": false,
      "verified_grasp_observed": false,
      "first_grasp_step": null,
      "first_verified_grasp_step": null,
      "final_phase": "GRASP",
      "final_cube_body_pos": [
        0.010954649187624454,
        0.02276298962533474,
        0.820329239136969
      ],
      "final_gripper_qpos_state": [
        0.03999587893486023,
        -0.040000125765800476
      ],
      "gripper_qpos_native_error": null
    },
    {
      "descend_z_offset": 0.06,
      "success": false,
      "native_grasp_observed": false,
      "verified_grasp_observed": false,
      "first_grasp_step": null,
      "first_verified_grasp_step": null,
      "final_phase": "GRASP",
      "final_cube_body_pos": [
        0.010954649187624454,
        0.02276298962533474,
        0.8202339152957658
      ],
      "final_gripper_qpos_state": [
        0.03999859467148781,
        -0.039999742060899734
      ],
      "gripper_qpos_native_error": null
    }
  ]
}