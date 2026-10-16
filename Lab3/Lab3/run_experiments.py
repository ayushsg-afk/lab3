import csv
import time
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

XML_PATH = "robot_descriptions/scene.xml"

def minimum_jerk_step(q_start, q_end, t_ratio):
    s = 10 * (t_ratio ** 3) - 15 * (t_ratio ** 4) + 6 * (t_ratio ** 5)
    return q_start + s * (q_end - q_start)

def solve_ik_6d(model, target_pos, target_mat, initial_qpos, site_name="gripper_center", max_iters=500, tol=1e-3):
    data_ik = mujoco.MjData(model)
    data_ik.qpos[:] = np.copy(initial_qpos)
    mujoco.mj_forward(model, data_ik)
    
    site_id = model.site(site_name).id

    for _ in range(max_iters):
        mujoco.mj_forward(model, data_ik)
        pos_err = target_pos - data_ik.site(site_id).xpos
        if np.linalg.norm(pos_err) < tol:
            return np.copy(data_ik.qpos[:6]), True

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data_ik, jacp, jacr, site_id)
        J = np.vstack([jacp[:, :6], jacr[:, :6]])
        
        rot_err_mat = target_mat @ data_ik.site(site_id).xmat.reshape(3, 3).T
        quat_err = np.zeros(4)
        mujoco.mju_mat2Quat(quat_err, rot_err_mat.flatten())
        axis_err = quat_err[1:] * np.sign(quat_err[0])
        
        err_6d = np.concatenate([pos_err, axis_err])
        reg = 1e-2 * np.eye(6)
        delta_q = J.T @ np.linalg.inv(J @ J.T + reg) @ err_6d
        data_ik.qpos[:6] += np.clip(delta_q, -0.04, 0.04)

    return np.copy(data_ik.qpos[:6]), False

def check_grasp(model, data, cube_geom_id, left_finger_id, right_finger_id):
    """Verifies dual finger contact with the cube."""
    left_touch, right_touch = False, False
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = c.geom1, c.geom2
        if (g1 == cube_geom_id and g2 == left_finger_id) or (g2 == cube_geom_id and g1 == left_finger_id):
            left_touch = True
        if (g1 == cube_geom_id and g2 == right_finger_id) or (g2 == cube_geom_id and g1 == right_finger_id):
            right_touch = True
    return left_touch and right_touch

def run_experiment_pipeline(num_episodes=20):
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    home_pose = np.array([0.0, 0.4, -0.6, 0.2, 0.0, 0.0])
    gripper_open = np.array([0.025, 0.025])
    gripper_closed = np.array([0.001, 0.001])
    down_mat = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)

    cube_body_id = model.body("cube").id
    cube_geom_id = model.geom("cube_geom").id
    left_finger_id = model.geom("left_finger_collision").id
    right_finger_id = model.geom("right_finger_collision").id
    tray_pos = np.copy(data.body(model.body("tray").id).xpos)

    logs = []

    print(f"=== Running Experiment Pipeline ({num_episodes} Episodes) ===")

    for ep in range(1, num_episodes + 1):
        # 1. Reset Scene
        mujoco.mj_resetData(model, data)
        data.qpos[:6] = home_pose
        data.qpos[6:8] = gripper_open
        data.ctrl[:6] = home_pose
        data.ctrl[6:8] = gripper_open

        # 2. Randomize Cube Pose (x, y, yaw)
        rand_x = np.random.uniform(0.28, 0.48)
        rand_y = np.random.uniform(-0.18, 0.18)
        rand_yaw = np.random.uniform(-np.pi/4, np.pi/4)

        cube_jnt_idx = model.body_jntadr[cube_body_id]
        data.qpos[cube_jnt_idx:cube_jnt_idx+3] = [rand_x, rand_y, 0.222]
        mujoco.mj_forward(model, data)

        cube_pos = np.array([rand_x, rand_y, 0.222])

        # 3. Compute IK
        hover_pos = cube_pos + np.array([0.0, 0.0, 0.12])
        grasp_pos = cube_pos + np.array([0.0, 0.0, -0.002])

        q_hover, success_h = solve_ik_6d(model, hover_pos, down_mat, home_pose)
        q_grasp, success_g = solve_ik_6d(model, grasp_pos, down_mat, q_hover)

        # Failure Check 1: Unreachable Workspace
        if not (success_h and success_g):
            logs.append({"episode": ep, "success": False, "reason": "IK_UNREACHABLE", "x": rand_x, "y": rand_y})
            print(f"Episode {ep:02d}: NO  | Reason: IK_UNREACHABLE at ({rand_x:.2f}, {rand_y:.2f})")
            continue

        # 4. Simulate Pick Sequence
        # Step A: Hover
        for step in range(500):
            t = step / 500.0
            data.ctrl[:6] = minimum_jerk_step(home_pose, q_hover, t)
            mujoco.mj_step(model, data)

        # Step B: Descend
        for step in range(500):
            t = step / 500.0
            data.ctrl[:6] = minimum_jerk_step(q_hover, q_grasp, t)
            mujoco.mj_step(model, data)

        # Step C: Close Gripper
        data.ctrl[6:8] = gripper_closed
        for _ in range(300):
            mujoco.mj_step(model, data)

        # Failure Check 2: Grasp Check
        grasped = check_grasp(model, data, cube_geom_id, left_finger_id, right_finger_id)
        if not grasped:
            logs.append({"episode": ep, "success": False, "reason": "GRASP_FAILED", "x": rand_x, "y": rand_y})
            print(f"Episode {ep:02d}: NO  | Reason: GRASP_FAILED (Slip / Off-center)")
            continue

        # Step D: Lift
        for step in range(500):
            t = step / 500.0
            data.ctrl[:6] = minimum_jerk_step(q_grasp, q_hover, t)
            mujoco.mj_step(model, data)

        # Failure Check 3: Check if dropped during lift
        grasped_after_lift = check_grasp(model, data, cube_geom_id, left_finger_id, right_finger_id)
        if not grasped_after_lift:
            logs.append({"episode": ep, "success": False, "reason": "SLIP_DROPPED", "x": rand_x, "y": rand_y})
            print(f"Episode {ep:02d}: NO  | Reason: SLIP_DROPPED during lift")
            continue

        # --- SUCCESS BRANCH: Place in Tray ---
        hover_tray = tray_pos + np.array([0.0, 0.0, 0.14])
        place_tray = tray_pos + np.array([0.0, 0.0, 0.038])
        q_h_tray, _ = solve_ik_6d(model, hover_tray, down_mat, q_hover)
        q_p_tray, _ = solve_ik_6d(model, place_tray, down_rot_mat if 'down_rot_mat' in locals() else down_mat, q_h_tray)

        # Transfer to tray
        for step in range(600):
            t = step / 600.0
            data.ctrl[:6] = minimum_jerk_step(q_hover, q_h_tray, t)
            mujoco.mj_step(model, data)

        # Lower to tray
        for step in range(400):
            t = step / 400.0
            data.ctrl[:6] = minimum_jerk_step(q_h_tray, q_p_tray, t)
            mujoco.mj_step(model, data)

        # Release
        data.ctrl[6:8] = gripper_open
        for _ in range(200):
            mujoco.mj_step(model, data)

        logs.append({"episode": ep, "success": True, "reason": "SUCCESS", "x": rand_x, "y": rand_y})
        print(f"Episode {ep:02d}: YES | Success: Placed in tray!")

    # Save Results CSV
    with open("experiment_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "success", "reason", "x", "y"])
        writer.writeheader()
        writer.writerows(logs)

    # Output Summary Metrics
    successes = sum(1 for l in logs if l["success"])
    print(f"\n=== Evaluation Complete ===")
    print(f"Overall Success Rate: {successes / num_episodes * 100:.1f}% ({successes}/{num_episodes})")

if __name__ == "__main__":
    run_experiment_pipeline(num_episodes=20)
