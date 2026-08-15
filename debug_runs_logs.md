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
      "success": true,
      "native_grasp_observed": true,
      "verified_grasp_observed": true,
      "first_grasp_step": 28,
      "first_verified_grasp_step": 28,
      "final_phase": "LIFT",
      "final_cube_body_pos": [
        0.0081251004046335,
        0.022674156700209773,
        0.8472203166736518
      ],
      "final_gripper_qpos_state": [
        0.024850958958268166,
        -0.02517806738615036
      ],
      "gripper_qpos_native_error": null
    },
    {
      "descend_z_offset": 0.01,
      "success": true,
      "native_grasp_observed": true,
      "verified_grasp_observed": true,
      "first_grasp_step": 27,
      "first_verified_grasp_step": 27,
      "final_phase": "LIFT",
      "final_cube_body_pos": [
        0.008112646091253384,
        0.0226829715939495,
        0.8472640282734556
      ],
      "final_gripper_qpos_state": [
        0.024183906614780426,
        -0.02375614084303379
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
        0.01303179719605437,
        0.021862396885653897,
        0.8204196872056907
      ],
      "final_gripper_qpos_state": [
        0.0004978454089723527,
        -0.000501746719237417
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
        0.8200663003575975
      ],
      "final_gripper_qpos_state": [
        0.0005009702290408313,
        -0.0004985887208022177
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
        0.8203181428695815
      ],
      "final_gripper_qpos_state": [
        0.000497164495754987,
        -0.0005024161073379219
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
        0.820563068910626
      ],
      "final_gripper_qpos_state": [
        0.0004948321147821844,
        -0.0005047970917075872
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
        0.8207003157696073
      ],
      "final_gripper_qpos_state": [
        0.0004950882284902036,
        -0.0005044923746027052
      ],
      "gripper_qpos_native_error": null
    }
  ]
}