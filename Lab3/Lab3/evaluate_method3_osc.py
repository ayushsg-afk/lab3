import time
import numpy as np
import mujoco
import mujoco.viewer

XML_PATH = "robot_descriptions/scene.xml"

def svd_damped_pseudoinverse(J, lmbda=1e-2):
    """Computes regularized inverse using SVD decomposition from notes."""
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    V = Vt.T
    
    # S_damped = S / (S^2 + lambda^2)
    S_damped = S / (S**2 + lmbda**2)
    
    # J_dls = V * diag(S_damped) * U^T
    return V @ np.diag(S_damped) @ U.T

def main():
    print("=== Method 3: Operational Space Control via SVD Inverse ===")
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    
    home_pose = np.array([0.0, 0.4, -0.8, 0.4, 0.0, 0.0])
    target_pos = np.array([0.38, 0.0, 0.225]) + np.array([0.0, 0.0, 0.12])
    target_quat = np.array([0.0, 0.7071, 0.7071, 0.0])
    
    kp_pos = 20.0
    kp_ori = 5.0
    
    viewer = mujoco.viewer.launch_passive(model, data)
    
    mujoco.mj_resetData(model, data)
    data.qpos[:6] = home_pose
    data.ctrl[:6] = home_pose
    mujoco.mj_forward(model, data)
    
    site_id = model.site("gripper_center").id
    
    for _ in range(1000):
        step_start = time.time()
        
        cur_pos = data.site(site_id).xpos
        cur_mat = data.site(site_id).xmat
        cur_quat = np.zeros(4)
        mujoco.mju_mat2Quat(cur_quat, cur_mat)
        
        # Calculate task error
        pos_err = target_pos - cur_pos
        
        w1, x1, y1, z1 = target_quat
        w2, x2, y2, z2 = np.array([cur_quat[0], -cur_quat[1], -cur_quat[2], -cur_quat[3]])
        q_err = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
        ori_err = 2.0 * q_err[1:]
        
        v_task = np.concatenate([kp_pos * pos_err, kp_ori * ori_err])
        
        # Calculate Jacobian
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        J = np.vstack([jacp[:, :6], jacr[:, :6]])
        
        # Solve inverse using SVD formulation
        J_pinv = svd_damped_pseudoinverse(J, lmbda=1e-2)
        
        # Joint velocity command with null-space centering
        q_curr = data.qpos[:6]
        dq_null = (np.eye(6) - J_pinv @ J) @ (1.0 * (home_pose - q_curr))
        dq_cmd = J_pinv @ v_task + dq_null
        
        # Continuous integration step
        data.ctrl[:6] = q_curr + dq_cmd * model.opt.timestep
        mujoco.mj_step(model, data)
        viewer.sync()
        
        elapsed = time.time() - step_start
        if model.opt.timestep > elapsed:
            time.sleep(model.opt.timestep - elapsed)
            
    viewer.close()

if __name__ == "__main__":
    main()
