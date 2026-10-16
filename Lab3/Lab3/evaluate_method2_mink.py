import time
import warnings
import csv
import numpy as np
import mujoco
import mujoco.viewer
import mink

warnings.filterwarnings("ignore", category=UserWarning, module="glfw")

XML_PATH = "robot_descriptions/scene.xml"

def solve_mink_ik(configuration, target_pos, target_mat, max_iters=200):
    """Solves inverse kinematics using Mink configuration and DAQP solver."""
    site_id = configuration.model.site("gripper_center").id
    
    # Task setup: Position and Frame tasks
    pos_task = mink.PositionTask(
        link_name="gripper_center",
        target_pos=target_pos,
        weight=2.0
    )
    
    ori_task = mink.FrameTask(
        link_name="gripper_center",
        target_orientation=target_mat,
        weight=0.1
    )
    
    posture_task = mink.PostureTask(
        model=configuration.model,
        cost=0.01
    )
    posture_task.set_target(np.array([0.0, 0.4, -0.8, 0.4, 0.0, 0.0]))

    tasks = [pos_task, ori_task, posture_task]
    limits = [mink.ConfigurationLimit(configuration.model)]

    iters = 0
    for i in range(max_iters):
        iters = i + 1
        configuration.update()
        
        cur_pos = configuration.data.site(site_id).xpos
        pos_err = np.linalg.norm(target_pos - cur_pos)
        
        if pos_err < 1e-3:
            return configuration.q, True, iters, pos_err

        # Solve differential QP step
        vel = mink.solve_ik(
            configuration=configuration,
            tasks=tasks,
            dt=0.002,
            solver="daqp",
            limits=limits
        )
        configuration.integrate_inplace(vel, dt=0.002)

    cur_pos = configuration.data.site(site_id).xpos
    final_err = np.linalg.norm(target_pos - cur_pos)
    return configuration.q, final_err < 2e-3, iters, final_err

def main():
    print("=== Running Method 2: Mink QP IK 25-Episode Benchmark ===")
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    
    home_pose = np.array([0.0, 0.4, -0.8, 0.4, 0.0, 0.0])
    base_down_mat = np.array([
        [0,  1,  0],
        [1,  0,  0],
        [0,  0, -1]
    ], dtype=np.float64)

    np.random.seed(42)
    results = []
    
    viewer = mujoco.viewer.launch_passive(model, data)
    
    try:
        for ep in range(25):
            mujoco.mj_resetData(model, data)
            data.qpos[:6] = home_pose
            data.ctrl[:6] = home_pose
            
            # Randomize cube position
            rand_x = np.random.uniform(0.32, 0.42)
            rand_y = np.random.uniform(-0.08, 0.08)
            cube_pos = np.array([rand_x, rand_y, 0.225])
            
            cube_body_id = model.body("cube").id
            cube_jnt_idx = model.body_jntadr[cube_body_id]
            data.qpos[cube_jnt_idx:cube_jnt_idx+7] = [cube_pos[0], cube_pos[1], cube_pos[2], 1, 0, 0, 0]
            mujoco.mj_forward(model, data)
            
            hover_pos = cube_pos + np.array([0.0, 0.0, 0.12])
            
            # Setup Mink Configuration
            configuration = mink.Configuration(model)
            configuration.update(home_pose)
            
            q_hover, success, iters, err = solve_mink_ik(configuration, hover_pos, base_down_mat)
            
            results.append({
                "episode": ep + 1,
                "success": 1 if success else 0,
                "iterations": iters,
                "error_mm": err * 1000.0
            })
            
            # Motion execution phase
            q_start = np.copy(data.qpos[:6])
            for step in range(30):
                t = step / 30.0
                s = 10*(t**3) - 15*(t**4) + 6*(t**5)
                data.ctrl[:6] = q_start + s * (q_hover[:6] - q_start)
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(model.opt.timestep)
                
            print(f"Episode {ep+1:02d}/25 | Success: {success} | Iterations: {iters:02d} | Error: {err*1000:.3f} mm")

    finally:
        viewer.close()

    # Save benchmark CSV log
    with open("method2_mink_25episodes.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "success", "iterations", "error_mm"])
        writer.writeheader()
        writer.writerows(results)

    success_rate = np.mean([r["success"] for r in results]) * 100
    mean_iters = np.mean([r["iterations"] for r in results])
    mean_err = np.mean([r["error_mm"] for r in results])
    print(f"\nMethod 2 Summary (25 Episodes): Success Rate = {success_rate:.1f}% | Avg Iterations = {mean_iters:.1f} | Avg Error = {mean_err:.3f} mm")

if __name__ == "__main__":
    main()
