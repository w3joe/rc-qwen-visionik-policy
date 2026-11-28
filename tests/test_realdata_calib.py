from __future__ import annotations

import numpy as np

from vlm_local.classical import cell, kinematics as K, planner as P


def test_sim_and_real_cell_modes() -> None:
    # 1. Simulator proprio stays strictly on nominal simulator calibration and gripper scale
    sim_ee, sim_rot = K.fk_pose_sim(K.HOME_Q)
    sim_obs = {
        "proprio": {
            "joint_pos": K.HOME_Q.tolist(),
            "joint_vel": [0.0] * 6,
            "gripper": 0.9968,
            "ee_pos": (sim_ee - cell.BASE_WORLD).tolist(),
            "ee_quat": K.mat_to_quat(sim_rot).tolist(),
        }
    }
    assert cell.detect_real_cell_from_obs(sim_obs) is False
    assert cell.is_real_cell() is False
    pr_sim = P._extract_proprio(sim_obs)
    assert np.allclose(pr_sim["ee_pos"], sim_ee)
    assert abs(pr_sim["gripper"] - 0.9968) < 1e-9
    assert K.clearance_penalty(K.HOME_Q) == 0.0

    # 2. Real Spark proprio auto-activates PnP camera poses, +157.5 deg bracket mount, FK TCP, and gripper scaling
    real_q = np.array([-2.932064, -1.279027, -2.076174, -1.365518, 1.570568, 0.614902], dtype=np.float64)
    real_obs = {
        "proprio": {
            "joint_pos": real_q.tolist(),
            "joint_vel": [0.0] * 6,
            "gripper": 0.3084,  # real M30 nut across-flats reading
            "ee_pos": [0.419275, -0.064810, 0.505264],  # real ROS URDF TCP (151.3 mm from gripperframe)
            "ee_quat": [0.504674, -0.501634, -0.494293, 0.499341],
        }
    }
    assert cell.detect_real_cell_from_obs(real_obs) is True
    assert cell.is_real_cell() is True
    pr_real = P._extract_proprio(real_obs)
    fk_ee_real, _ = K.fk_pose(real_q)
    assert np.allclose(pr_real["ee_pos"], fk_ee_real)
    assert pr_real["gripper"] > P._min_grip_for_part("nut_m30")
    assert cell.normalize_gripper_reading(0.009, real_cell=True) < P.EMPTY_GRIPPER_THRESH
    assert cell.normalize_gripper_reading(0.107, real_cell=True) > P.EMPTY_GRIPPER_THRESH

    # 3. Spark negative-elbow branch clearance & IK continuity
    assert K.clearance_penalty(K.SPARK_HOME_Q) == 0.0
    ik = K.BoundedDampedIK()
    for target in (
        np.array([0.75, 0.10, 0.25]),
        np.array([0.60, 0.08, 0.18]),
        np.array([0.67, -0.07, 0.28]),
    ):
        sol = ik.solve(target, K.down_mat(0.0), K.SPARK_HOME_Q)
        assert sol.converged and sol.pos_err < 1e-3 and sol.clearance_pen == 0.0
        assert abs(K.wrap_angle(float(sol.q[0] - K.SPARK_HOME_Q[0]))) < 0.85
        assert sol.q[2] < -0.5

    # Restore sim auto-state cleanly
    assert cell.detect_real_cell_from_obs(sim_obs) is False
