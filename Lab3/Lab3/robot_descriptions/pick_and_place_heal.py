import time
import numpy as np
import mujoco
import mujoco.viewer

XML_PATH = "robot_descriptions/scene.xml"

def minimum_jerk_step(q_start, q_end, t_ratio):
    s = 10 * (t_ratio ** 3) - 15 * (t_ratio ** 4) + 6 * (t_ratio ** 5)
    return q_start + s * (q_end - q_start)

def solve_ik_6d(model, target_pos, target_mat, initial_qpos, site_name="gripper_center", max_iters=500, tol=1e-3, damping=1e-2):
    data_ik = mujoco.MjData(model)
    data_ik.qpos[:] = np.copy(initial_qpos)
    mujoco.mj_forward(model, data_ik)
    
    site_id = model.site(site_name).id

    for _ in range(max_iters):
        mujoco.mj_forward(model, data_ik)
        
        current_pos = data_ik.site(site_id).xpos
        pos_err = target_pos - current_pos
        
        current_mat = data_ik.site(site_id).xmat.reshape(3, 3)
        rot_err_mat = target_mat @ current_mat.T
        
        quat_err = np.zeros(4)
        mujoco.mju_mat2Quat(quat_err, rot_err_mat.flatten())
        axis_err = quat_err[1:] * np.sign(quat_err[0])
        
        err_6d = np.concatenate([pos_err, axis_err])
        
        if np.linalg.norm(pos_err) < tol and np.linalg.norm(axis_err) < 0.02:
            break

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data_ik, jacp, jacr, site_id)
        
        J_6d = np.vstack([jacp[:, :6], jacr[:, :6]])
        reg = damping * np.eye(6)
        delta_q = J_6d.T @ np.linalg.inv(J_6d @ J_6d.T + reg) @ err_6d
        data_ik.qpos[:6] += np.clip(delta_q, -0.03, 0.03)

    return np.copy(data_ik.qpos[:6])

def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    home_arm_joints = np.array([0.0, 0.3, -0.5, 0.2, 0.0, 0.0])
    gripper_open = np.array([0.025, 0.025])
    gripper_closed = np.array([0.002, 0.002])  # Soft contact grip limit

    data.qpos[:6] = home_arm_joints
    data.qpos[6:8] = gripper_open
    data.ctrl[:6] = home_arm_joints
    data.ctrl[6:8] = gripper_open
    mujoco.mj_forward(model, data)

    cube_id = model.body("cube").id
    tray_id = model.body("tray").id
    
    cube_pos = np.copy(data.body(cube_id).xpos)
    tray_pos = np.copy(data.body(tray_id).xpos)

    down_rot_mat = np.array([
        [ 1,  0,  0],
        [ 0, -1,  0],
        [ 0,  0, -1]
    ], dtype=np.float64)

    hover_cube_pos = cube_pos + np.array([0.0, 0.0, 0.12])
    grasp_cube_pos = cube_pos + np.array([0.0, 0.0, 0.0])
    hover_tray_pos = tray_pos + np.array([0.0, 0.0, 0.14])
    place_tray_pos = tray_pos + np.array([0.0, 0.0, 0.04])

    print("[INFO] Computing Stable Waypoints...")
    q_hover_cube = solve_ik_6d(model, hover_cube_pos, down_rot_mat, data.qpos)
    q_grasp_cube = solve_ik_6d(model, grasp_cube_pos, down_rot_mat, data.qpos)
    q_hover_tray = solve_ik_6d(model, hover_tray_pos, down_rot_mat, data.qpos)
    q_place_tray = solve_ik_6d(model, place_tray_pos, down_rot_mat, data.qpos)

    sequence = [
        (q_hover_cube, gripper_open, 2.5, "[INFO] Move to hover target"),
        (q_grasp_cube, gripper_open, 2.0, "[INFO] Descending to grasp position"),
        (q_grasp_cube, gripper_closed, 1.5, "[INFO] Closing gripper gently"),
        (q_hover_cube, gripper_closed, 2.0, "[INFO] Lifting object cleanly"),
        (q_hover_tray, gripper_closed, 3.0, "[INFO] Moving smoothly to tray"),
        (q_place_tray, gripper_closed, 1.8, "[INFO] Lowering into tray"),
        (q_place_tray, gripper_open, 1.2, "[INFO] Opening gripper - Dropping object"),
        (q_hover_tray, gripper_open, 1.5, "[INFO] Retracting arm"),
        (home_arm_joints, gripper_open, 2.5, "[INFO] Return home")
    ]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        current_arm = np.copy(home_arm_joints)
        current_gripper = np.copy(gripper_open)

        for _ in range(200):
            mujoco.mj_step(model, data)
            viewer.sync()

        print("--- Executing Smooth & Stable Pick-and-Place ---")

        for target_arm, target_gripper, duration, log_msg in sequence:
            print(log_msg)
            steps = int(duration / model.opt.timestep)
            start_arm = np.copy(current_arm)
            start_gripper = np.copy(current_gripper)

            for step in range(steps):
                step_start = time.time()
                t_ratio = step / float(steps)

                arm_ctrl = minimum_jerk_step(start_arm, target_arm, t_ratio)
                gripper_ctrl = minimum_jerk_step(start_gripper, target_gripper, t_ratio)

                data.ctrl[:6] = arm_ctrl
                data.ctrl[6:8] = gripper_ctrl

                mujoco.mj_step(model, data)
                viewer.sync()

                elapsed = time.time() - step_start
                if model.opt.timestep > elapsed:
                    time.sleep(model.opt.timestep - elapsed)

            current_arm = np.copy(target_arm)
            current_gripper = np.copy(target_gripper)

        print("--- Stable Pick-and-Place Completed Successfully ---")

        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

if __name__ == "__main__":
    main()

