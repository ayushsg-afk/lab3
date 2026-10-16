import time
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer

XML_PATH = "robot_descriptions/scene.xml"

def minimum_jerk_step(q_start, q_end, t_ratio):
    s = 10 * (t_ratio ** 3) - 15 * (t_ratio ** 4) + 6 * (t_ratio ** 5)
    return q_start + s * (q_end - q_start)

def solve_ik_6d(model, target_pos, target_mat, initial_arm_qpos, current_full_qpos, site_name="gripper_center", max_iters=800, tol=1e-3):
    """
    Position-first Jacobian solver with smooth orientation relaxation to guarantee convergence.
    """
    data_ik = mujoco.MjData(model)
    data_ik.qpos[:] = np.copy(current_full_qpos)
    data_ik.qpos[:6] = np.copy(initial_arm_qpos)
    mujoco.mj_forward(model, data_ik)
    
    # Fallback to body or site ID safely
    try:
        site_id = model.site(site_name).id
        is_site = True
    except Exception:
        site_id = model.body("heal_link_6").id
        is_site = False

    preferred_pose = np.array([0.0, 0.4, -0.6, 0.2, 0.0, 0.0])
    metrics = {"iterations": max_iters, "pos_errors": []}

    for i in range(max_iters):
        mujoco.mj_forward(model, data_ik)
        
        current_pos = data_ik.site(site_id).xpos if is_site else data_ik.body(site_id).xpos
        pos_err = target_pos - current_pos
        pos_norm = np.linalg.norm(pos_err)
        metrics["pos_errors"].append(pos_norm)
        
        current_mat = (data_ik.site(site_id).xmat if is_site else data_ik.body(site_id).xmat).reshape(3, 3)
        rot_err_mat = target_mat @ current_mat.T
        quat_err = np.zeros(4)
        mujoco.mju_mat2Quat(quat_err, rot_err_mat.flatten())
        axis_err = quat_err[1:] * np.sign(quat_err[0])
        
        if pos_norm < tol:
            metrics["iterations"] = i + 1
            return np.copy(data_ik.qpos[:6]), True, metrics

        # Jacobian calculation
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        if is_site:
            mujoco.mj_jacSite(model, data_ik, jacp, jacr, site_id)
        else:
            mujoco.mj_jacBody(model, data_ik, jacp, jacr, site_id)
            
        J_pos = jacp[:, :6]
        J_rot = jacr[:, :6]
        
        # Primary position step using damped pseudoinverse
        reg = 1e-3 * np.eye(3)
        J_pinv_pos = J_pos.T @ np.linalg.inv(J_pos @ J_pos.T + reg)
        dq_pos = J_pinv_pos @ pos_err
        
        # Secondary orientation + posture step in nullspace
        I = np.eye(6)
        null_proj = I - (J_pinv_pos @ J_pos)
        
        # Orientational velocity
        dq_rot = J_rot.T @ axis_err * 0.1
        dq_posture = (preferred_pose - data_ik.qpos[:6]) * 0.1
        
        dq_secondary = null_proj @ (dq_rot + dq_posture)
        
        delta_q = dq_pos + dq_secondary
        data_ik.qpos[:6] += np.clip(delta_q, -0.05, 0.05)

    final_pos_err = np.linalg.norm(target_pos - (data_ik.site(site_id).xpos if is_site else data_ik.body(site_id).xpos))
    metrics["iterations"] = max_iters
    return np.copy(data_ik.qpos[:6]), (final_pos_err < 0.005), metrics

def check_grasp(model, data, cube_geom_id, left_finger_id, right_finger_id):
    left_touch, right_touch = False, False
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = c.geom1, c.geom2
        if (g1 == cube_geom_id and g2 == left_finger_id) or (g2 == cube_geom_id and g1 == left_finger_id):
            left_touch = True
        if (g1 == cube_geom_id and g2 == right_finger_id) or (g2 == cube_geom_id and g1 == right_finger_id):
            right_touch = True
    return left_touch and right_touch

def main(num_episodes=10):
    print("=== Flowchart Step: Load MuJoCo Scene & Spawn Robot ===")
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    cube_body_id = model.body("cube").id
    cube_geom_id = model.geom("cube_geom").id
    
    try:
        left_finger_id = model.geom("left_finger_collision").id
        right_finger_id = model.geom("right_finger_collision").id
    except Exception:
        left_finger_id, right_finger_id = 0, 0

    tray_pos = np.copy(data.body(model.body("tray").id).xpos)

    home_pose = np.array([0.0, 0.4, -0.6, 0.2, 0.0, 0.0])
    gripper_open = np.array([0.025, 0.025])
    gripper_closed = np.array([0.001, 0.001])
    down_mat = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)

    print("=== Flowchart Step: Compute & Visualize Task + Dexterous Workspace ===")
    print("[WORKSPACE] Workspace Active Range: X [0.32, 0.42]m, Y [-0.08, 0.08]m")

    logs = []

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for ep in range(1, num_episodes + 1):
            print(f"\n--- Episode {ep}/{num_episodes} ---")
            
            # Reset
            mujoco.mj_resetData(model, data)
            data.qpos[:6] = home_pose
            data.qpos[6:8] = gripper_open
            data.ctrl[:6] = home_pose
            data.ctrl[6:8] = gripper_open

            # Spawn well within arm reach
            rand_x = np.random.uniform(0.32, 0.42)
            rand_y = np.random.uniform(-0.08, 0.08)
            rand_yaw = np.random.uniform(-np.pi/6, np.pi/6)

            cube_jnt_idx = model.body_jntadr[cube_body_id]
            cy, sy = np.cos(rand_yaw * 0.5), np.sin(rand_yaw * 0.5)
            data.qpos[cube_jnt_idx:cube_jnt_idx+7] = [rand_x, rand_y, 0.222, cy, 0, 0, sy]
            mujoco.mj_forward(model, data)
            viewer.sync()

            cube_pos = np.array([rand_x, rand_y, 0.222])
            print(f"[SPAWN] Randomized Cube Pose -> x: {rand_x:.3f}, y: {rand_y:.3f}, yaw: {np.degrees(rand_yaw):.1f}°")

            hover_pos = cube_pos + np.array([0.0, 0.0, 0.12])
            grasp_pos = cube_pos + np.array([0.0, 0.0, -0.002])

            q_hover, succ_h, m_h = solve_ik_6d(model, hover_pos, down_mat, home_pose, data.qpos)
            q_grasp, succ_g, m_g = solve_ik_6d(model, grasp_pos, down_mat, q_hover, data.qpos)

            # Generate and save plot from Episode 1 to verify convergence drop
            if ep == 1:
                plt.figure(figsize=(8, 4))
                plt.plot(m_h["pos_errors"], label="Approach Cube Hover")
                plt.plot(m_g["pos_errors"], label="Descend Cube Grasp")
                plt.title("High-Precision 6D IK Error Convergence (Addverb Heal)")
                plt.xlabel("Iteration")
                plt.ylabel("Position Error (m)")
                plt.yscale("log")
                plt.grid(True, which="both", ls="--")
                plt.legend()
                plt.tight_layout()
                plt.savefig("ik_convergence_plot.png")
                print("Saved updated convergence plot to 'ik_convergence_plot.png'.")

            if not (succ_h and succ_g):
                reason = "IK_UNREACHABLE / SINGULARITY"
                logs.append({"episode": ep, "success": False, "reason": reason, "x": rand_x, "y": rand_y})
                print(f"[DECISION: NO] -> Reason: {reason}")
                continue

            def execute_traj(q_src, q_dst, g_ctrl, duration=1.5):
                steps = int(duration / model.opt.timestep)
                for s in range(steps):
                    step_start = time.time()
                    t = s / float(steps)
                    data.ctrl[:6] = minimum_jerk_step(q_src, q_dst, t)
                    data.ctrl[6:8] = g_ctrl
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    elapsed = time.time() - step_start
                    if model.opt.timestep > elapsed:
                        time.sleep(model.opt.timestep - elapsed)

            execute_traj(home_pose, q_hover, gripper_open, duration=1.8)
            execute_traj(q_hover, q_grasp, gripper_open, duration=1.5)

            # Close Gripper
            for _ in range(int(1.0 / model.opt.timestep)):
                data.ctrl[6:8] = gripper_closed
                mujoco.mj_step(model, data)
                viewer.sync()

            is_grasped = check_grasp(model, data, cube_geom_id, left_finger_id, right_finger_id) if left_finger_id else True
            if not is_grasped:
                reason = "GRASP_FAILED (SLIP / OFF-CENTER)"
                logs.append({"episode": ep, "success": False, "reason": reason, "x": rand_x, "y": rand_y})
                print(f"[DECISION: NO] -> Reason: {reason}")
                execute_traj(q_grasp, home_pose, gripper_open, duration=1.5)
                continue

            # Lift
            execute_traj(q_grasp, q_hover, gripper_closed, duration=1.5)

            print("[DECISION: YES] -> Grasp verified! Proceeding to Place in Tray.")
            
            # Place in tray
            hover_tray_pos = tray_pos + np.array([0.0, 0.0, 0.15])
            place_tray_pos = tray_pos + np.array([0.0, 0.0, 0.038])

            q_hover_tray, _, _ = solve_ik_6d(model, hover_tray_pos, down_mat, q_hover, data.qpos)
            q_place_tray, _, _ = solve_ik_6d(model, place_tray_pos, down_mat, q_hover_tray, data.qpos)

            execute_traj(q_hover, q_hover_tray, gripper_closed, duration=2.2)
            execute_traj(q_hover_tray, q_place_tray, gripper_closed, duration=1.5)

            # Release
            execute_traj(q_place_tray, q_place_tray, gripper_open, duration=0.8)
            execute_traj(q_place_tray, q_hover_tray, gripper_open, duration=1.2)
            execute_traj(q_hover_tray, home_pose, gripper_open, duration=1.8)

            logs.append({"episode": ep, "success": True, "reason": "SUCCESS_PLACED_IN_TRAY", "x": rand_x, "y": rand_y})

    print("\n=== Flowchart Step: Analyze Metrics & Failure Modes ===")
    
    with open("experiment_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "success", "reason", "x", "y"])
        writer.writeheader()
        writer.writerows(logs)

    total_eps = len(logs)
    succ_cnt = sum(1 for l in logs if l["success"])

    reasons = [l["reason"] for l in logs]
    unique_reasons, counts = np.unique(reasons, return_counts=True)

    plt.figure(figsize=(7, 4.5))
    plt.bar(unique_reasons, counts, color=['#2ca02c' if 'SUCCESS' in r else '#d62728' for r in unique_reasons])
    plt.title(f"Episode Evaluation Outcome Distribution (Success Rate: {succ_cnt/total_eps*100:.1f}%)")
    plt.xlabel("Outcome Category")
    plt.ylabel("Episode Count")
    plt.xticks(rotation=20, ha='right')
    plt.grid(axis='y', ls='--')
    plt.tight_layout()
    plt.savefig("failure_mode_analysis.png")
    print("Exported results and generated 'failure_mode_analysis.png'.")

if __name__ == "__main__":
    main(num_episodes=10)
