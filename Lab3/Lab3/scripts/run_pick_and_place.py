import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer

# Path to your scene.xml
XML_PATH = os.path.abspath("robot_descriptions/scene.xml")


def solve_dls_ik(model, data, target_pos, target_quat, site_id, damping=0.05, max_step=0.05):
    """Damped Least Squares IK solver for 6-DOF HEAL arm."""
    tcp_pos = data.site_xpos[site_id]
    tcp_mat = data.site_xmat[site_id].reshape(3, 3)

    pos_err = target_pos - tcp_pos

    tgt_w, tgt_x, tgt_y, tgt_z = target_quat
    tgt_mat = np.array([
        [1 - 2*(tgt_y**2 + tgt_z**2), 2*(tgt_x*tgt_y - tgt_z*tgt_w), 2*(tgt_x*tgt_z + tgt_y*tgt_w)],
        [2*(tgt_x*tgt_y + tgt_z*tgt_w), 1 - 2*(tgt_x**2 + tgt_z**2), 2*(tgt_y*tgt_z - tgt_x*tgt_w)],
        [2*(tgt_x*tgt_z - tgt_y*tgt_w), 2*(tgt_y*tgt_z + tgt_x*tgt_w), 1 - 2*(tgt_x**2 + tgt_y**2)]
    ])

    rot_err_mat = tgt_mat @ tcp_mat.T
    rot_err = 0.5 * np.array([
        rot_err_mat[2, 1] - rot_err_mat[1, 2],
        rot_err_mat[0, 2] - rot_err_mat[2, 0],
        rot_err_mat[1, 0] - rot_err_mat[0, 1]
    ])

    error_6d = np.hstack([pos_err, rot_err])

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    J = np.vstack([jacp[:, :6], jacr[:, :6]])

    n = J.shape[0]
    J_dls = J.T @ np.linalg.pinv(J @ J.T + (damping ** 2) * np.eye(n))
    dq = J_dls @ error_6d
    return np.clip(dq, -max_step, max_step)


def main():
    if not os.path.exists(XML_PATH):
        raise FileNotFoundError(f"Could not find XML file at {XML_PATH}")

    print(f"[INFO] Loading scene directly from: {XML_PATH}")
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # Make arm material visible programmatically
    mat_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "arm")
    if mat_id >= 0:
        model.mat_rgba[mat_id, 3] = 1.0

    # End Effector Site ID
    site_id = model.site("right_center").id

    # Actuator IDs for Gripper
    left_act = model.actuator("left_finger_act").id
    right_act = model.actuator("right_finger_act").id

    # Home posture
    data.qpos[:6] = np.array([0.0, -0.4, 0.8, 0.0, 0.4, 0.0])
    mujoco.mj_forward(model, data)

    # Waypoints based on table/cube positions in scene.xml
    cube_body_id = model.body("cube").id
    
    target_quat = np.array([0.0, 0.0, 1.0, 0.0])
    phase = "pregrasp"

    print("[INFO] Launching MuJoCo Passive Viewer...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        time.sleep(0.5)
        while viewer.is_running():
            cube_pos = data.xpos[cube_body_id]
            
            pregrasp_pos  = np.array([cube_pos[0], cube_pos[1], cube_pos[2] + 0.15])
            grasp_pos     = np.array([cube_pos[0], cube_pos[1], cube_pos[2] + 0.02])
            lift_pos      = np.array([cube_pos[0], cube_pos[1], cube_pos[2] + 0.25])
            tray_drop_pos = np.array([0.54, 0.40, 0.35])

            if phase == "pregrasp":
                target_pos = pregrasp_pos
                data.ctrl[left_act] = 0.025
                data.ctrl[right_act] = 0.025
            elif phase == "grasp":
                target_pos = grasp_pos
                data.ctrl[left_act] = 0.0
                data.ctrl[right_act] = 0.0
            elif phase == "lift":
                target_pos = lift_pos
                data.ctrl[left_act] = 0.0
                data.ctrl[right_act] = 0.0
            elif phase == "tray_drop":
                target_pos = tray_drop_pos
                data.ctrl[left_act] = 0.0
                data.ctrl[right_act] = 0.0
            elif phase == "release":
                target_pos = tray_drop_pos
                data.ctrl[left_act] = 0.025
                data.ctrl[right_act] = 0.025

            # IK step
            dq = solve_dls_ik(model, data, target_pos, target_quat, site_id)
            data.qpos[:6] += dq
            data.qfrc_applied[:6] = data.qfrc_bias[:6]

            tcp_pos = data.site_xpos[site_id]
            pos_err = np.linalg.norm(target_pos - tcp_pos)

            if phase == "pregrasp" and pos_err < 0.02:
                phase = "grasp"
            elif phase == "grasp" and pos_err < 0.015:
                phase = "lift"
            elif phase == "lift" and pos_err < 0.025:
                phase = "tray_drop"
            elif phase == "tray_drop" and pos_err < 0.025:
                phase = "release"

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
