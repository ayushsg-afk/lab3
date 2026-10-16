import time
import numpy as np
import mujoco
import mujoco.viewer

XML_PATH = "robot_descriptions/franka_scene.xml"

def minimum_jerk_step(q_start, q_end, t_ratio):
    """5th-order minimum jerk trajectory polynomial."""
    s = 10 * (t_ratio ** 3) - 15 * (t_ratio ** 4) + 6 * (t_ratio ** 5)
    return q_start + s * (q_end - q_start)

def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # Initial Ready Pose (Arm raised to clear the table)
    home_pose = np.array([0.0, -0.5, 0.0, -1.5, 0.0, 1.8, 0.785, 0.035, 0.035])

    # 9-DOF Target Waypoints
    waypoints = [
        # 1. Rotate base & elevate over red cube
        (np.array([-0.25, -0.2, 0.0, -1.6, 0.0, 2.0, 0.785, 0.035, 0.035]), 2.0),
        # 2. Descend over cube
        (np.array([-0.25, 0.25, 0.0, -2.1, 0.0, 2.5, 0.785, 0.035, 0.035]), 1.8),
        # 3. Close gripper fingers to grip cube
        (np.array([-0.25, 0.25, 0.0, -2.1, 0.0, 2.5, 0.785, 0.001, 0.001]), 1.0),
        # 4. Lift cube clear of table
        (np.array([-0.25, -0.3, 0.0, -1.5, 0.0, 1.8, 0.785, 0.001, 0.001]), 2.0),
        # 5. Swing over blue tray
        (np.array([0.35, -0.3, 0.0, -1.5, 0.0, 1.8, 0.785, 0.001, 0.001]), 2.5),
        # 6. Lower into blue tray
        (np.array([0.35, 0.20, 0.0, -2.0, 0.0, 2.3, 0.785, 0.001, 0.001]), 1.8),
        # 7. Release cube
        (np.array([0.35, 0.20, 0.0, -2.0, 0.0, 2.3, 0.785, 0.035, 0.035]), 1.0),
        # 8. Retract back to home
        (home_pose, 2.5),
    ]

    # Initialize positions
    data.qpos[:7] = home_pose[:7]
    data.ctrl[:9] = home_pose
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        current_pose = np.copy(home_pose)

        for _ in range(150):
            mujoco.mj_step(model, data)
            viewer.sync()

        print("--- Executing Franka 7-DOF Pick and Place ---")

        for target_pose, duration in waypoints:
            steps = int(duration / model.opt.timestep)

            for step in range(steps):
                step_start = time.time()
                t_ratio = step / float(steps)

                ctrl_target = minimum_jerk_step(current_pose, target_pose, t_ratio)
                data.ctrl[:9] = ctrl_target

                mujoco.mj_step(model, data)
                viewer.sync()

                elapsed = time.time() - step_start
                if model.opt.timestep > elapsed:
                    time.sleep(model.opt.timestep - elapsed)

            current_pose = np.copy(target_pose)

        print("--- Pick and Place Sequence Finished ---")

        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

if __name__ == "__main__":
    main()
