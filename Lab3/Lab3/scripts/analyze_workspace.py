import numpy as np
import matplotlib.pyplot as plt
import mujoco

def analyze_workspace():
    model = mujoco.MjModel.from_xml_path("robot_descriptions/single_arm_heal_effort_actuation_rs_mj.xml")
    data = mujoco.MjData(model)

    site_id = model.site("right_center").id
    num_samples = 30000

    task_pts = []
    dexterous_pts = []

    print("Sampling workspace...")
    for _ in range(num_samples):
        q = np.random.uniform(-np.pi, np.pi, size=6)
        data.qpos[:6] = q
        mujoco.mj_kinematics(model, data)

        pos = data.site_xpos[site_id].copy()
        rot = data.site_xmat[site_id].reshape(3, 3)

        # Dexterous condition: End-effector pointing downward (Z-axis alignment)
        is_downward = rot[2, 2] < -0.85
        
        task_pts.append(pos)
        if is_downward:
            dexterous_pts.append(pos)

    task_pts = np.array(task_pts)
    dexterous_pts = np.array(dexterous_pts)

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(task_pts[:, 0], task_pts[:, 1], task_pts[:, 2], s=0.5, c='navy', alpha=0.1)
    ax1.set_title("Task Workspace (Reachable TCP)")
    ax1.set_xlabel("X (m)"); ax1.set_ylabel("Y (m)"); ax1.set_zlabel("Z (m)")

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(dexterous_pts[:, 0], dexterous_pts[:, 1], dexterous_pts[:, 2], s=0.5, c='forestgreen', alpha=0.1)
    ax2.set_title("Dexterous Workspace (Top-Down Grasping)")
    ax2.set_xlabel("X (m)"); ax2.set_ylabel("Y (m)"); ax2.set_zlabel("Z (m)")

    plt.tight_layout()
    plt.savefig("report/workspace_analysis.png", dpi=300)
    print("Workspace plot successfully saved to report/workspace_analysis.png")

if __name__ == "__main__":
    analyze_workspace()
