import numpy as np
import mujoco

def solve_dls_ik(model, data, target_pos, target_quat, site_id, lambda_reg=1e-3):
    # Position tracking error
    pos_err = target_pos - data.site_xpos[site_id]

    # Orientation tracking error
    curr_quat = np.zeros(4)
    mujoco.mju_mat2Quat(curr_quat, data.site_xmat[site_id])

    if np.dot(target_quat, curr_quat) < 0:
        curr_quat = -curr_quat

    w_t, v_t = target_quat[0], target_quat[1:]
    w_c, v_c = curr_quat[0], curr_quat[1:]
    rot_err = w_c * v_t - w_t * v_c - np.cross(v_t, v_c)

    # Task space tracking step vector (3D pos, 3D rot)
    x_step = np.concatenate([pos_err * 0.5, rot_err * 0.2])

    # Extract site Jacobian
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    J = np.vstack([jacp[:, :6], jacr[:, :6]])

    # Closed-form DLS solve: delta_q = (J^T J + lambda I)^(-1) J^T x_step
    A = J.T @ J + lambda_reg * np.eye(6)
    b = J.T @ x_step
    dq = np.linalg.solve(A, b)

    return np.clip(dq, -0.08, 0.08)
