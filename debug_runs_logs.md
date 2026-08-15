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
      "first_grasp_step": 29,
      "first_verified_grasp_step": 29,
      "final_phase": "LIFT",
      "final_cube_body_pos": [
        0.008576852674632315,
        0.022963804677485326,
        0.8477461663923875
      ],
      "final_gripper_qpos_state": [
        0.02386479079723358,
        -0.02408423274755478
      ],
      "gripper_qpos_native_error": null
    },
    {
      "descend_z_offset": 0.01,
      "success": true,
      "native_grasp_observed": true,
      "verified_grasp_observed": true,
      "first_grasp_step": 26,
      "first_verified_grasp_step": 26,
      "final_phase": "LIFT",
      "final_cube_body_pos": [
        0.008207396482088989,
        0.02278615189263526,
        0.8472842716659009
      ],
      "final_gripper_qpos_state": [
        0.025909721851348877,
        -0.025728169828653336
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
        0.012161224769691736,
        0.021432872684770825,
        0.8203253071870785
      ],
      "final_gripper_qpos_state": [
        0.0004971513408236206,
        -0.0005024309502914548
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
        0.8213848604915419
      ],
      "final_gripper_qpos_state": [
        0.0004933337331749499,
        -0.000506273761857301
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
        0.8207794970903624
      ],
      "final_gripper_qpos_state": [
        0.0004939762293361127,
        -0.0005056015797890723
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
        0.8215107748214268
      ],
      "final_gripper_qpos_state": [
        0.0004957038327120245,
        -0.0005038672825321555
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
        0.8207252831092323
      ],
      "final_gripper_qpos_state": [
        0.0004985750420019031,
        -0.0005009936867281795
      ],
      "gripper_qpos_native_error": null
    }
  ]
}