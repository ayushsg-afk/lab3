import time
import numpy as np
import mujoco
import mujoco.viewer

# Path relative to the scripts folder
SCENE_XML = "../robot_descriptions/franka_scene.xml"

def solve_dls_ik(model, data, site_name, target_pos, damping=1e-2):
    """Computes joint velocity step using Damped Least Squares Inverse Kinematics."""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id == -1:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, site_name)

    current_pos = data.site_xpos[site_id] if site_id < model.nsite else data.xpos[site_id]
    pos_err = target_pos - current_pos

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

    J = jacp[:, :6]
    n = J.shape[0]
    A = J @ J.T + (damping ** 2) * np.eye(n)
    dq = J.T @ np.linalg.solve(A, pos_err)

    return dq

def run_pick_and_place():
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)

    ee_site = "attachment_site"
    cube_body = "cube"
    tray_body = "tray"

    mujoco.mj_step(model, data)

    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cube_body)
    tray_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, tray_body)

    cube_pos = data.xpos[cube_id].copy()
    tray_pos = data.xpos[tray_id].copy()

    pre_grasp_pos = cube_pos + np.array([0.0, 0.0, 0.15])
    grasp_pos     = cube_pos + np.array([0.0, 0.0, 0.02])
    lift_pos      = cube_pos + np.array([0.0, 0.0, 0.20])
    tray_above    = tray_pos + np.array([0.0, 0.0, 0.15])
    drop_pos      = tray_pos + np.array([0.0, 0.0, 0.05])

    phases = ["pregrasp", "approach", "close_gripper", "lift", "transit", "drop_approach", "open_gripper", "done"]
    phase_idx = 0
    phase = phases[phase_idx]

    print("[INFO] Starting Pick and Place Loop...")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        phase_start = time.time()

        while viewer.is_running():
            step_start = time.time()
            elapsed_phase = time.time() - phase_start

            if phase == "pregrasp":
                target = pre_grasp_pos
                gripper_cmd = 0.04
                if elapsed_phase > 2.0:
                    phase_idx += 1
                    phase = phases[phase_idx]
                    phase_start = time.time()
                    print("[INFO] Reached pregrasp position")

            elif phase == "approach":
                target = grasp_pos
                gripper_cmd = 0.04
                if elapsed_phase > 2.0:
                    phase_idx += 1
                    phase = phases[phase_idx]
                    phase_start = time.time()
                    print("[INFO] Approached cube")

            elif phase == "close_gripper":
                target = grasp_pos
                gripper_cmd = 0.00
                if elapsed_phase > 1.5:
                    phase_idx += 1
                    phase = phases[phase_idx]
                    phase_start = time.time()
                    print("[INFO] Closed gripper")

            elif phase == "lift":
                target = lift_pos
                gripper_cmd = 0.00
                if elapsed_phase > 2.0:
                    phase_idx += 1
                    phase = phases[phase_idx]
                    phase_start = time.time()
                    print("[INFO] Lifted object")

            elif phase == "transit":
                target = tray_above
                gripper_cmd = 0.00
                if elapsed_phase > 2.5:
                    phase_idx += 1
                    phase = phases[phase_idx]
                    phase_start = time.time()
                    print("[INFO] Transited to tray")

            elif phase == "drop_approach":
                target = drop_pos
                gripper_cmd = 0.00
                if elapsed_phase > 1.5:
                    phase_idx += 1
                    phase = phases[phase_idx]
                    phase_start = time.time()
                    print("[INFO] Lowered into tray")

            elif phase == "open_gripper":
                target = drop_pos
                gripper_cmd = 0.04
                if elapsed_phase > 1.5:
                    phase_idx += 1
                    phase = phases[phase_idx]
                    phase_start = time.time()
                    print("[INFO] Opened gripper, task complete!")

            elif phase == "done":
                target = tray_above
                gripper_cmd = 0.04

            dq = solve_dls_ik(model, data, ee_site, target)
            data.ctrl[:6] = data.qpos[:6] + dq * 0.05
            if model.nu > 6:
                data.ctrl[6] = gripper_cmd

            mujoco.mj_step(model, data)
            viewer.sync()

            time_until_next_frame = model.opt.timestep - (time.time() - step_start)
            if time_until_next_frame > 0:
                time.sleep(time_until_next_frame)

if __name__ == "__main__":
    run_pick_and_place()
