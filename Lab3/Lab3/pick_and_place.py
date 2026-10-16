import time
import numpy as np
import mujoco
import mujoco.viewer

XML_PATH = "robot_descriptions/scene.xml"

def minimum_jerk_step(q_start, q_end, t_ratio):
    s = 10 * (t_ratio ** 3) - 15 * (t_ratio ** 4) + 6 * (t_ratio ** 5)
    return q_start + s * (q_end - q_start)

def solve_dls_ik(model, data, target_pos, target_mat, home_pose, site_name="gripper_center", max_iters=250):
    # Always seed IK from the forward-facing home posture
    q = np.copy(home_pose)
    damping = 1e-2
    k_null = 2.0  # Strong null-space weight to keep elbow pointing up and arm forward
    
    d_ik = mujoco.MjData(model)
    
    for i in range(max_iters):
        d_ik.qpos[:6] = q
        mujoco.mj_forward(model, d_ik)
        
        try:
            site_id = model.site(site_name).id
            current_pos = d_ik.site(site_id).xpos
            current_mat = d_ik.site(site_id).xmat.reshape(3, 3)
        except Exception:
            body_id = model.body("heal_link_6").id
            current_pos = d_ik.body(body_id).xpos
            current_mat = d_ik.body(body_id).xmat.reshape(3, 3)
            
        pos_err = target_pos - current_pos
        
        if np.linalg.norm(pos_err) < 1e-3:
            return q, True

        R_err = target_mat @ current_mat.T
        axis_err = 0.5 * np.array([
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1]
        ])
        
        e_6d = np.concatenate([pos_err, axis_err])
        
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        site_names = [model.site(idx).name for idx in range(model.nsite)]
        if site_name in site_names:
            mujoco.mj_jacSite(model, d_ik, jacp, jacr, model.site(site_name).id)
        else:
            mujoco.mj_jacBody(model, d_ik, jacp, jacr, body_id)
            
        J = np.vstack([jacp[:, :6], jacr[:, :6]])
        
        J_dls = J.T @ np.linalg.inv(J @ J.T + (damping ** 2) * np.eye(6))
        dq_task = J_dls @ e_6d
        
        I = np.eye(6)
        dq_null = (I - J_dls @ J) @ (k_null * (home_pose - q))
        
        q += dq_task + dq_null
        
        # Strictly lock base joint (Joint 0) to stay facing forward towards the table
        q[0] = 0.0
        
    return q, False

def main():
    print("=== Running Forward-Reaching Pick and Place Demo ===")
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    cube_body_id = model.body("cube").id
    tray_pos = np.copy(data.body(model.body("tray").id).xpos)

    # Clean natural forward-reaching elbow pose over table
    home_pose = np.array([0.0, 0.5, -1.0, 0.5, 1.57, 0.0])
    
    gripper_open = np.array([0.025, 0.025])
    gripper_closed = np.array([0.0, 0.0])
    
    # End-effector orientation: fingers pointing down, wrist aligned forward (+X)
    target_rot_mat = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [-1, 0, 0]
    ], dtype=np.float64)

    viewer = mujoco.viewer.launch_passive(model, data)

    try:
        # Reset scene
        mujoco.mj_resetData(model, data)
        data.qpos[:6] = home_pose
        data.qpos[6:8] = gripper_open
        data.ctrl[:6] = home_pose
        data.ctrl[6:8] = gripper_open
        mujoco.mj_forward(model, data)

        rand_x, rand_y = 0.38, 0.0
        cube_jnt_idx = model.body_jntadr[cube_body_id]
        data.qpos[cube_jnt_idx:cube_jnt_idx+7] = [rand_x, rand_y, 0.225, 1, 0, 0, 0]
        mujoco.mj_forward(model, data)
        viewer.sync()

        cube_pos = np.array([rand_x, rand_y, 0.225])

        # Precise waypoints over table
        hover_pos = cube_pos + np.array([0.0, 0.0, 0.12])
        grasp_pos = cube_pos + np.array([0.0, 0.0, -0.010])

        hover_tray_pos = tray_pos + np.array([0.0, 0.0, 0.15])
        place_tray_pos = tray_pos + np.array([0.0, 0.0, 0.038])

        # Solve IK for each target
        q_hover, _ = solve_dls_ik(model, data, hover_pos, target_rot_mat, home_pose)
        q_grasp, _ = solve_dls_ik(model, data, grasp_pos, target_rot_mat, home_pose)
        q_hover_tray, _ = solve_dls_ik(model, data, hover_tray_pos, target_rot_mat, home_pose)
        q_place_tray, _ = solve_dls_ik(model, data, place_tray_pos, target_rot_mat, home_pose)

        def move_smoothly(q_target, g_ctrl, duration=2.0):
            q_start = np.copy(data.qpos[:6])
            steps = int(duration / model.opt.timestep)
            for s in range(steps):
                step_start = time.time()
                t = s / float(steps)
                data.ctrl[:6] = minimum_jerk_step(q_start, q_target, t)
                data.ctrl[6:8] = g_ctrl
                mujoco.mj_step(model, data)
                viewer.sync()
                elapsed = time.time() - step_start
                if model.opt.timestep > elapsed:
                    time.sleep(model.opt.timestep - elapsed)

        print("[1] Moving smoothly forward over block...")
        move_smoothly(q_hover, gripper_open, duration=2.5)

        print("[2] Descending around block...")
        move_smoothly(q_grasp, gripper_open, duration=2.0)

        print("[3] Closing gripper fingers...")
        q_stat = np.copy(data.qpos[:6])
        for _ in range(int(1.5 / model.opt.timestep)):
            data.ctrl[:6] = q_stat
            data.ctrl[6:8] = gripper_closed
            mujoco.mj_step(model, data)
            viewer.sync()

        print("[4] Lifting block...")
        move_smoothly(q_hover, gripper_closed, duration=2.0)

        print("[5] Moving to tray...")
        move_smoothly(q_hover_tray, gripper_closed, duration=2.5)

        print("[6] Lowering inside tray...")
        move_smoothly(q_place_tray, gripper_closed, duration=2.0)

        print("[7] Releasing block...")
        q_stat_tray = np.copy(data.qpos[:6])
        for _ in range(int(1.0 / model.opt.timestep)):
            data.ctrl[:6] = q_stat_tray
            data.ctrl[6:8] = gripper_open
            mujoco.mj_step(model, data)
            viewer.sync()

        print("[8] Returning home...")
        move_smoothly(q_hover_tray, gripper_open, duration=1.5)
        move_smoothly(home_pose, gripper_open, duration=2.5)

        print("=== Success! Block placed in tray cleanly. ===")

    finally:
        viewer.close()

if __name__ == "__main__":
    main()
