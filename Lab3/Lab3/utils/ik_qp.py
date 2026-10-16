import numpy as np
import mujoco
from scipy.optimize import lsq_linear

def solve_qp_ik(model, data, target_pos, target_quat, site_id, lambda_reg=1e-3):
    pos_err = target_pos - data.site_xpos[site_id]

    curr_quat = np.zeros(4)
    mujoco.mju_mat2Quat(curr_quat, data.site_xmat[site_id])

    if np.dot(target_quat, curr_quat) < 0:
        curr_quat = -curr_quat

    w_t, v_t = target_quat[0], target_quat[1:]
    w_c, v_c = curr_quat[0], curr_quat[1:]
    rot_err = w_c * v_t - w_t * v_c - np.cross(v_t, v_c)

    x_step = np.concatenate([pos_err * 0.5, rot_err * 0.2])

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    J = np.vstack([jacp[:, :6], jacr[:, :6]])

    # Joint limit delta bounds
    q_curr = data.qpos[:6]
    q_min = model.jnt_range[:6, 0]
    q_max = model.jnt_range[:6, 1]

    max_step = 0.08
    lb = np.array([max(-max_step, q_min[i] - q_curr[i]) for i in range(6)])
    ub = np.array([min(max_step, q_max[i] - q_curr[i]) for i in range(6)])

    for i in range(6):
        if lb[i] > ub[i]:
            lb[i] = ub[i] = 0.0

    res = lsq_linear(J, x_step, bounds=(lb, ub), lsmr_tol='auto')
    
    return res.x if res.success else np.zeros(6)
