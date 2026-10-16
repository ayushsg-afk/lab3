cat << 'EOF' > ~/Downloads/Lab3/Lab3/evaluate_method1_dls.py
import time
import warnings
import csv
import numpy as np
import mujoco
import mujoco.viewer

warnings.filterwarnings("ignore", category=UserWarning, module="glfw")

XML_PATH = "robot_descriptions/scene.xml"

def solve_dls_ik(model, data, target_pos, target_mat, q_seed, max_iters=300, lmbda=1e-3):
    q = np.copy(q_seed)
    d_ik = mujoco.MjData(model)
    iters = 0
    k_null = 0.2
    
    for i in range(max_iters):
        iters = i + 1
        d_ik.qpos[:6] = q
        mujoco.mj_forward(model, d_ik)
        
        try:
            site_id = model.site("gripper_center").id
            cur_pos = d_ik.site(site_id).xpos
            cur_mat = d_ik.site(site_id).xmat.reshape(3, 3)
        except Exception:
            body_id = model.body("heal_link_6").id
            cur_pos = d_ik.body(body_id).xpos
            cur_mat = d_ik.body(body_id).xmat.reshape(3, 3)
            
        pos_err = target_pos - cur_pos
        
        # Orientation error
        R_err = target_mat @ cur_mat.T
        ori_err = 0.5 * np.array([
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1]
        ])
        
        pos_norm = np.linalg.norm(pos_err)
        
        if pos_norm < 1e-3:
            return q, True, iters, pos_norm

        # Task space error with higher weight on translational accuracy
        e_6d = np.concatenate([pos_err * 2.0, ori_err * 0.1])

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        site_names = [model.site(idx).name for idx in range(model.nsite)]
        if "gripper_center" in site_names:
            mujoco.mj_jacSite(model, d_ik, jacp, jacr, model.site("gripper_center").id)
        else:
            mujoco.mj_jacBody(model, d_ik, jacp, jacr, model.body("heal_link_6").id)
            
        J = np.vstack([jacp[:, :6], jacr[:, :6]])
        
        JT = J.T
        H = JT @ J + (lmbda ** 2) * np.eye(6)
        J_dls = np.linalg.solve(H, JT)
        
        dq_task = J_dls @ e_6d
        q_home = np.array([0.0, 0.4, -0.8, 0.4, 0.0, 0.0])
        dq_null = (np.eye(6) - J_dls @ J) @ (k_null * (q_home - q))
        
        dq = dq_task + dq_null
        
        # Limit max step size to keep IK stable
        max_step = 0.15
        if np.linalg.norm(dq) > max_step:
            dq = dq * (max_step / np.linalg.norm(dq))
            
        q += dq
        
    return q, False, iters, pos_norm

def main():
    print("=== Running Method 1: DLS-IK 25-Episode Benchmark (Fixed Pose Math) ===")
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    
    home_pose = np.array([0.0, 0.4, -0.8, 0.4, 0.0, 0.0])
    
    # Downward orientation matrix
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
            
            rand_x = np.random.uniform(0.32, 0.42)
            rand_y = np.random.uniform(-0.08, 0.08)
            cube_pos = np.array([rand_x, rand_y, 0.225])
            
            cube_body_id = model.body("cube").id
            cube_jnt_idx = model.body_jntadr[cube_body_id]
            data.qpos[cube_jnt_idx:cube_jnt_idx+7] = [cube_pos[0], cube_pos[1], cube_pos[2], 1, 0, 0, 0]
            mujoco.mj_forward(model, data)
            
            hover_pos = cube_pos + np.array([0.0, 0.0, 0.12])
            q_hover, success, iters, err = solve_dls_ik(model, data, hover_pos, base_down_mat, home_pose)
            
            results.append({
                "episode": ep + 1,
                "success": 1 if success else 0,
                "iterations": iters,
                "error_mm": err * 1000.0
            })
            
            q_start = np.copy(data.qpos[:6])
            for step in range(30):
                t = step / 30.0
                s = 10*(t**3) - 15*(t**4) + 6*(t**5)
                data.ctrl[:6] = q_start + s * (q_hover - q_start)
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(model.opt.timestep)
                
            print(f"Episode {ep+1:02d}/25 | Success: {success} | Iterations: {iters} | Error: {err*1000:.3f} mm")

    finally:
        viewer.close()

    with open("method1_dls_25episodes.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "success", "iterations", "error_mm"])
        writer.writeheader()
        writer.writerows(results)

    success_rate = np.mean([r["success"] for r in results]) * 100
    mean_iters = np.mean([r["iterations"] for r in results])
    mean_err = np.mean([r["error_mm"] for r in results])
    print(f"\nMethod 1 Summary (25 Episodes): Success Rate = {success_rate:.1f}% | Avg Iterations = {mean_iters:.1f} | Avg Error = {mean_err:.3f} mm")

if __name__ == "__main__":
    main()
EOF
