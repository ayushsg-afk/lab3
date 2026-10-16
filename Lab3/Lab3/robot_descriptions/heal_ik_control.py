#!/usr/bin/env python3
"""
HEAL 6-DOF - Standalone Interactive IK & FK Controller with GUI.

Two windows open:
  * MuJoCo viewer           : Displays the live HEAL arm
  * "HEAL Control Panel"    : Joint Sliders (FK) + Cartesian End-Effector Sliders (IK)

Usage:
    PYTHONPATH=. python3 heal_ik_control.py
"""
import argparse
import os
import time
import numpy as np
import mujoco
import mujoco.viewer

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:
    raise SystemExit("Tkinter missing. Install it with: sudo apt install -y python3-tk")

# Locate robot_descriptions/ directory regardless of execution folder
_here = os.path.dirname(os.path.abspath(__file__))
for _cand in (_here, os.path.join(_here, ".."), os.getcwd()):
    if os.path.isdir(os.path.join(_cand, "robot_descriptions")):
        ROOT = os.path.abspath(_cand)
        break
else:
    raise SystemExit("robot_descriptions/ folder not found - place this script in project root.")

RD = os.path.join(ROOT, "robot_descriptions")

# Preferred XML model selection
XML_CANDIDATES = [
    os.path.join(RD, "single_arm_heal_effort_actuation_rs_mj_2.xml"),
    os.path.join(RD, "arm_gripper.xml"),
    os.path.join(RD, "world.xml")
]

XML = next((path for path in XML_CANDIDATES if os.path.exists(path)), None)
if XML is None:
    raise SystemExit(f"None of the candidate XML files found in {RD}")

NAME, DOF = "HEAL", 6
JOINTS = [f"joint_{i}" for i in range(1, 7)]
HOME = [0.0, 0.5, 0.5, 0.0, 0.5, 0.0]


def fix_model(model):
    """Ensures geometry visibility if transparency is set to zero."""
    m = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "arm")
    if m >= 0:
        model.mat_rgba[m, 3] = 1.0


def jid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def bid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def sid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)


def load_model():
    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)
    fix_model(model)
    return model, data


# ---------- SE(3) Transformations ----------
def euler_to_quat(roll, pitch, yaw):
    """Converts Roll-Pitch-Yaw angles (radians) to Quaternion [w, x, y, z]."""
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z])


def quat_to_euler(quat):
    """Converts Quaternion [w, x, y, z] to Roll-Pitch-Yaw angles (radians)."""
    w, x, y, z = quat
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


# ---------- Damped Least Squares Inverse Kinematics ----------
def solve_dls_ik(model, data, target_pos, target_quat, site_id, damping=0.05, max_step=0.1):
    """Calculates joint angle displacement dq using Damped Least Squares IK."""
    tcp_pos = data.site_xpos[site_id]
    tcp_mat = data.site_xmat[site_id].reshape(3, 3)

    # Position Error
    pos_err = target_pos - tcp_pos

    # Orientation Error via Quaternion Difference
    tgt_w, tgt_x, tgt_y, tgt_z = target_quat
    tgt_mat = np.array([
        [1 - 2*(tgt_y**2 + tgt_z**2), 2*(tgt_x*tgt_y - tgt_z*tgt_w), 2*(tgt_x*tgt_z + tgt_y*tgt_w)],
        [2*(tgt_x*tgt_y + tgt_z*tgt_w), 1 - 2*(tgt_x**2 + tgt_z**2), 2*(tgt_y*tgt_z - tgt_x*tgt_w)],
        [2*(tgt_x*tgt_z - tgt_y*tgt_w), 2*(tgt_y*tgt_z + tgt_x*tgt_w), 1 - 2*(tgt_x**2 + tgt_y**2)]
    ])
    
    rot_err_mat = tgt_mat @ tcp_mat.T
    rot_err = 0.5 * np.array([
        rot_err_mat[2, 1] - rot_err_mat[1, 2],
        rot_err_mat[0, 2] - rot_err_mat[2, 0],
        rot_err_mat[1, 0] - rot_err_mat[0, 1]
    ])

    # 6D Spatial Error Vector
    error_6d = np.hstack([pos_err, rot_err])

    # Compute Jacobian matrix (3D pos + 3D rot)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    J = np.vstack([jacp[:, :6], jacr[:, :6]])

    # Damped Least Squares inversion: J_dls = J^T * (J * J^T + lambda^2 * I)^(-1)
    n = J.shape[0]
    J_dls = J.T @ np.linalg.pinv(J @ J.T + (damping ** 2) * np.eye(n))
    dq = J_dls @ error_6d

    # Clip step size for stability
    dq = np.clip(dq, -max_step, max_step)
    return dq


# ---------- Main Initialization ----------
model, data = load_model()

# Identify Joint Limits & Addresses
lo = np.array([model.jnt_range[jid(model, n), 0] for n in JOINTS])
hi = np.array([model.jnt_range[jid(model, n), 1] for n in JOINTS])
q_addr = [model.jnt_qposadr[jid(model, n)] for n in JOINTS]

# End-Effector site identification
site_name = "right_center" if sid(model, "right_center") >= 0 else ("tcp" if sid(model, "tcp") >= 0 else None)
if site_name:
    site_id = sid(model, site_name)
else:
    site_id = 0

# Set initial pose
home_q = np.clip(np.array(HOME, float), lo, hi)
data.qpos[q_addr] = home_q
mujoco.mj_forward(model, data)

curr_tcp_pos = np.copy(data.site_xpos[site_id])
curr_tcp_quat = np.array([1.0, 0.0, 0.0, 0.0])
curr_r, curr_p, curr_y = quat_to_euler(curr_tcp_quat)

# ------------------------------------------------------------------ GUI SETUP
root = tk.Tk()
root.title("HEAL 6-DOF IK & FK Control")

notebook = ttk.Notebook(root)
notebook.pack(padx=10, pady=5, fill="both", expand=True)

tab_ik = ttk.Frame(notebook)
tab_fk = ttk.Frame(notebook)
notebook.add(tab_ik, text=" Inverse Kinematics (IK) ")
notebook.add(tab_fk, text=" Forward Kinematics (FK) ")

# --- IK Controls (Cartesian EE Position & Orientation) ---
ik_vars = {
    'x': tk.DoubleVar(value=float(curr_tcp_pos[0])),
    'y': tk.DoubleVar(value=float(curr_tcp_pos[1])),
    'z': tk.DoubleVar(value=float(curr_tcp_pos[2])),
    'roll': tk.DoubleVar(value=float(np.degrees(curr_r))),
    'pitch': tk.DoubleVar(value=float(np.degrees(curr_p))),
    'yaw': tk.DoubleVar(value=float(np.degrees(curr_y)))
}

ik_specs = [
    ('x', 'Position X (m)', -0.8, 0.8, 0.01),
    ('y', 'Position Y (m)', -0.8, 0.8, 0.01),
    ('z', 'Position Z (m)', 0.0, 1.2, 0.01),
    ('roll', 'Roll (deg)', -180.0, 180.0, 1.0),
    ('pitch', 'Pitch (deg)', -180.0, 180.0, 1.0),
    ('yaw', 'Yaw (deg)', -180.0, 180.0, 1.0)
]

for key, label_str, f_from, f_to, res in ik_specs:
    tk.Scale(tab_ik, variable=ik_vars[key], from_=f_from, to=f_to, resolution=res,
             orient=tk.HORIZONTAL, length=400, label=label_str).pack(padx=10, pady=2)

# --- FK Controls (Joint Angle Sliders) ---
fk_vars = []
for i, name in enumerate(JOINTS):
    v = tk.DoubleVar(value=float(np.degrees(home_q[i])))
    fk_vars.append(v)
    tk.Scale(tab_fk, variable=v, from_=float(np.degrees(lo[i])), to=float(np.degrees(hi[i])),
             resolution=0.5, orient=tk.HORIZONTAL, length=400,
             label=f"{name} (deg)").pack(padx=10, pady=2)


def sync_fk_from_q(q_rad):
    """Updates FK joint slider displays when IK solver alters joint angles."""
    for v, qi in zip(fk_vars, q_rad):
        v.set(float(np.degrees(qi)))


def reset_to_home():
    data.qpos[q_addr] = home_q
    mujoco.mj_forward(model, data)
    sync_fk_from_q(home_q)
    
    tcp_p = data.site_xpos[site_id]
    ik_vars['x'].set(float(tcp_p[0]))
    ik_vars['y'].set(float(tcp_p[1]))
    ik_vars['z'].set(float(tcp_p[2]))


btn_frame = tk.Frame(root)
btn_frame.pack(pady=4)
tk.Button(btn_frame, text="Reset to Home Pose", command=reset_to_home).pack(side=tk.LEFT, padx=6)

info_lbl = tk.Label(root, text="", font=("DejaVu Sans Mono", 9), justify=tk.LEFT)
info_lbl.pack(padx=10, pady=6)

# ------------------------------------------------------------------ SIMULATION LOOP
viewer = mujoco.viewer.launch_passive(model, data)

def tick():
    if not viewer.is_running():
        root.destroy()
        return

    selected_tab = notebook.index(notebook.select())

    if selected_tab == 0:  # Active Tab: IK
        target_pos = np.array([ik_vars['x'].get(), ik_vars['y'].get(), ik_vars['z'].get()])
        r_rad = np.radians(ik_vars['roll'].get())
        p_rad = np.radians(ik_vars['pitch'].get())
        y_rad = np.radians(ik_vars['yaw'].get())
        target_quat = euler_to_quat(r_rad, p_rad, y_rad)

        # Iterate IK convergence steps
        for _ in range(5):
            dq = solve_dls_ik(model, data, target_pos, target_quat, site_id)
            data.qpos[q_addr] = np.clip(data.qpos[q_addr] + dq, lo, hi)
            mujoco.mj_forward(model, data)

        sync_fk_from_q(data.qpos[q_addr])

    else:  # Active Tab: FK
        q_rad = np.radians([v.get() for v in fk_vars])
        data.qpos[q_addr] = q_rad
        mujoco.mj_forward(model, data)

        # Update IK targets to reflect manual FK adjustments
        tcp_p = data.site_xpos[site_id]
        ik_vars['x'].set(float(tcp_p[0]))
        ik_vars['y'].set(float(tcp_p[1]))
        ik_vars['z'].set(float(tcp_p[2]))

    # Advance physics simulation step
    mujoco.mj_step(model, data)
    viewer.sync()

    # Calculate actual TCP status
    actual_pos = data.site_xpos[site_id]
    current_q_deg = np.degrees(data.qpos[q_addr])
    
    info_text = (
        f"TCP Position (XYZ) : [{actual_pos[0]:+.4f}, {actual_pos[1]:+.4f}, {actual_pos[2]:+.4f}] m\n"
        f"Joints (Q1 - Q6)   : [{', '.join([f'{q:+.1f}°' for q in current_q_deg])}]"
    )
    info_lbl.config(text=info_text)

    root.after(20, tick)


root.after(20, tick)
root.protocol("WM_DELETE_WINDOW", lambda: (viewer.close(), root.destroy()))
root.mainloop()
