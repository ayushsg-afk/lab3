import time
import numpy as np
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import mink

XML_PATH = "robot_descriptions/scene.xml"

def minimum_jerk_step(q_start, q_end, t_ratio):
    """5th-order minimum jerk trajectory polynomial."""
    s = 10 * (t_ratio ** 3) - 15 * (t_ratio ** 4) + 6 * (t_ratio ** 5)
    return q_start + s * (q_end - q_start)

def solve_mink_ik(configuration, target_pos, target_mat, frame_name="gripper_center", max_iters=200):
    """
    Solves 6D IK using Mink's Differential IK optimization.
    """
    # 1. End-Effector Frame Task (Position + Downward Orientation)
    transform_target = mink.SE3.from_rotation_and_translation(
        rotation=mink.SO3.from_matrix(target_mat),
        translation=target_pos
    )
    frame_task = mink.FrameTask(
        frame_name=frame_name,
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1e-3
    )
    frame_task.set_target(transform_target)

    # 2. Posture Task (Keep preferred high-elbow configuration in null-space)
    posture_task = mink.PostureTask(model=configuration.model, cost=1e-2)

    tasks = [frame_task, posture_task]
    limits = [mink.ConfigurationLimit(model=configuration.model)]
    
    solver = "osqp"
    dt = 1e-2
    metrics = {"iterations": max_iters, "pos_errors": []}

    for i in range(max_iters):
        # Calculate velocity vector using Mink solver with joint limits
        vel = mink.solve_ik(configuration, tasks, rate_limit=None, dt=dt, solver=solver, limits=limits, damp=1e-3)
        configuration.integrate_inplace(vel, dt)
        
        # Track position error
        ee_pos = configuration.get_sitenet_transform(frame_name).translation
        err = np.linalg.norm(target_pos - ee_pos)
        metrics["pos_errors"].append(err)
        
        if err < 5e-4:
            metrics["iterations"] = i + 1
            break

    return np.copy(configuration.q[:6]), metrics

def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # Wrap model into Mink Configuration
    configuration = mink.Configuration(model)

    home_arm_joints = np.array([0.0, 0.4, -0.6, 0.2, 0.0, 0.0])
    gripper_open = np.array([0.025, 0.025])
    gripper_closed = np.array([0.001, 0.001])

    data.qpos[:6] = home_arm_joints
    data.qpos[6:8] = gripper_open
    data.ctrl[:6] = home_arm_joints
    data.ctrl[6:8] = gripper_open
    mujoco.mj_forward(model, data)
    configuration.update(data.qpos)

    # Target World Coordinates
    cube_id = model.body("cube").id
    tray_id = model.body("tray").id
    
    cube_pos = np.copy(data.body(cube_id).xpos)
    tray_pos = np.copy(data.body(tray_id).xpos)

    # Downward orientation matrix
    down_rot_mat = np.array([
        [ 1,  0,  0],
        [ 0, -1,  0],
        [ 0,  0, -1]
    ], dtype=np.float64)

    # Waypoints
    hover_cube_pos = cube_pos + np.array([0.0, 0.0, 0.14])
    grasp_cube_pos = cube_pos + np.array([0.0, 0.0, -0.002])
    hover_tray_pos = tray_pos + np.array([0.0, 0.0, 0.15])
    place_tray_pos = tray_pos + np.array([0.0, 0.0, 0.038])

    print("=== Solving IK via Mink (Differential IK Optimization) ===")
    q_hover_cube, m1 = solve_mink_ik(configuration, hover_cube_pos, down_rot_mat)
    q_grasp_cube, m2 = solve_mink_ik(configuration, grasp_cube_pos, down_rot_mat)
    q_hover_tray, m3 = solve_mink_ik(configuration, hover_tray_pos, down_rot_mat)
    q_place_tray, m4 = solve_mink_ik(configuration, place_tray_pos, down_rot_mat)

    print(f"Mink Approach Hover IK: {m1['iterations']} iters, Error: {m1['pos_errors'][-1]:.6f} m")
    print(f"Mink Descend Grasp IK:  {m2['iterations']} iters, Error: {m2['pos_errors'][-1]:.6f} m")
    print(f"Mink Transfer Tray IK: {m3['iterations']} iters, Error: {m3['pos_errors'][-1]:.6f} m")
    print(f"Mink Lower Place IK:   {m4['iterations']} iters, Error: {m4['pos_errors'][-1]:.6f} m")

    # Plot convergence for lab report comparison
    plt.figure(figsize=(8, 4))
    plt.plot(m1["pos_errors"], label="Mink: Approach Cube")
    plt.plot(m2["pos_errors"], label="Mink: Descend Cube")
    plt.plot(m3["pos_errors"], label="Mink: Transfer Tray")
    plt.plot(m4["pos_errors"], label="Mink: Lower Place")
    plt.title("Mink Optimization IK Convergence (OSQP Solver)")
    plt.xlabel("Iteration")
    plt.ylabel("Position Error (m)")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig("mink_ik_convergence_plot.png")
    print("\nSaved 'mink_ik_convergence_plot.png' for comparative analysis report.")

    sequence = [
        (q_hover_cube, gripper_open, 2.5, "[MINK] Approach hover target"),
        (q_grasp_cube, gripper_open, 2.0, "[MINK] Descend over block"),
        (q_grasp_cube, gripper_closed, 1.2, "[MINK] Close gripper"),
        (q_hover_cube, gripper_closed, 2.0, "[MINK] Lift block"),
        (q_hover_tray, gripper_closed, 3.0, "[MINK] Move over tray"),
        (q_place_tray, gripper_closed, 1.8, "[MINK] Lower inside tray"),
        (q_place_tray, gripper_open, 1.2, "[MINK] Release block"),
        (q_hover_tray, gripper_open, 1.5, "[MINK] Retract upwards"),
        (home_arm_joints, gripper_open, 2.5, "[MINK] Return home")
    ]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        current_arm = np.copy(home_arm_joints)
        current_gripper = np.copy(gripper_open)

        for _ in range(150):
            mujoco.mj_step(model, data)
            viewer.sync()

        print("\n--- Executing Mink-Guided Pick and Place ---")

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

        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

if __name__ == "__main__":
    main()
