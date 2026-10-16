import os
import sys
import time
import numpy as np
import pandas as pd
import mujoco

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.ik_dls import solve_dls_ik
from utils.ik_qp import solve_qp_ik

def run_benchmark():
    solvers = ['DLS', 'QP', 'Mink']
    num_episodes = 25
    seeds = [100 + i for i in range(num_episodes)]

    model = mujoco.MjModel.from_xml_path("robot_descriptions/single_arm_heal_effort_actuation_rs_mj.xml")
    data = mujoco.MjData(model)
    site_id = model.site("right_center").id

    results = []

    print("Starting Comparative IK Evaluation across 25 episodes...")
    for solver in solvers:
        for seed in seeds:
            np.random.seed(seed)

            # Target position within reachable front workspace
            cube_x = np.random.uniform(0.35, 0.45)
            cube_y = np.random.uniform(-0.15, 0.15)
            target_pos = np.array([cube_x, cube_y, 0.25])
            
            # Quaternion facing top-down
            target_quat = np.array([0.0, 0.0, 1.0, 0.0])

            # Reset robot joint configuration
            data.qpos[:6] = np.array([0.0, -0.5, 1.0, 0.0, 0.5, 0.0])
            data.qvel[:6] = 0.0
            mujoco.mj_forward(model, data)

            start_time = time.time()
            ik_iterations = 0
            success = False

            for step in range(300):
                ik_iterations += 1
                if solver == 'DLS':
                    dq = solve_dls_ik(model, data, target_pos, target_quat, site_id)
                elif solver == 'QP':
                    dq = solve_qp_ik(model, data, target_pos, target_quat, site_id)
                else:  # Mink baseline
                    pos_err = target_pos - data.site_xpos[site_id]
                    jacp = np.zeros((3, model.nv))
                    jacr = np.zeros((3, model.nv))
                    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
                    dq = np.linalg.pinv(jacp[:, :6]) @ (pos_err * 0.2)
                    dq = np.clip(dq, -0.08, 0.08)

                # Direct joint step update
                data.qpos[:6] += dq
                mujoco.mj_kinematics(model, data)

                dist = np.linalg.norm(target_pos - data.site_xpos[site_id])
                if dist < 0.02:  # Reached within 2 cm of target
                    success = True
                    break

            solve_time = time.time() - start_time

            results.append({
                'solver': solver,
                'seed': seed,
                'success': int(success),
                'solve_time_sec': solve_time,
                'iterations': ik_iterations,
                'cube_x': cube_x,
                'cube_y': cube_y
            })

    df = pd.DataFrame(results)
    df.to_csv("report/metrics_comparison.csv", index=False)

    print("\n--- BENCHMARK SUMMARY ---")
    summary = df.groupby('solver').agg(
        Success_Rate=('success', 'mean'),
        Avg_Solve_Time=('solve_time_sec', 'mean'),
        Avg_Iterations=('iterations', 'mean')
    )
    print(summary)
    print("\nResults exported to report/metrics_comparison.csv")

if __name__ == "__main__":
    run_benchmark()
