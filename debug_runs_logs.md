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
      "first_grasp_step": 20,
      "final_phase": "DESCEND",
      "final_cube_body_pos": [
        0.0047136699071995735,
        0.018847792869961775,
        0.8205237282879919
      ],
      "gripper_qpos_native_error": null
    },
    {
      "descend_z_offset": 0.01,
      "success": false,
      "native_grasp_observed": true,
      "first_grasp_step": 22,
      "final_phase": "DESCEND",
      "final_cube_body_pos": [
        0.016990693970489256,
        0.023148433348870293,
        0.8159235810559772
      ],
      "gripper_qpos_native_error": null
    },
    {
      "descend_z_offset": 0.02,
      "success": false,
      "native_grasp_observed": false,
      "first_grasp_step": null,
      "final_phase": "GRASP",
      "final_cube_body_pos": [
        0.010895580286874046,
        0.02272727996625108,
        0.8208393490528483
      ],
      "gripper_qpos_native_error": null
    },
    {
      "descend_z_offset": 0.03,
      "success": false,
      "native_grasp_observed": false,
      "first_grasp_step": null,
      "final_phase": "GRASP",
      "final_cube_body_pos": [
        0.010954649187624454,
        0.02276298962533474,
        0.8203930336346641
      ],
      "gripper_qpos_native_error": null
    },
    {
      "descend_z_offset": 0.04,
      "success": false,
      "native_grasp_observed": false,
      "first_grasp_step": null,
      "final_phase": "GRASP",
      "final_cube_body_pos": [
        0.010954649187624454,
        0.02276298962533474,
        0.8217496645234665
      ],
      "gripper_qpos_native_error": null
    },
    {
      "descend_z_offset": 0.05,
      "success": false,
      "native_grasp_observed": false,
      "first_grasp_step": null,
      "final_phase": "GRASP",
      "final_cube_body_pos": [
        0.010954649187624454,
        0.02276298962533474,
        0.8199704248347425
      ],
      "gripper_qpos_native_error": null
    },
    {
      "descend_z_offset": 0.06,
      "success": false,
      "native_grasp_observed": false,
      "first_grasp_step": null,
      "final_phase": "GRASP",
      "final_cube_body_pos": [
        0.010954649187624454,
        0.02276298962533474,
        0.820334233601648
      ],
      "gripper_qpos_native_error": null
    }
  ]
}