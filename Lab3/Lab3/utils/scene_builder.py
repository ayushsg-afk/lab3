import os
import numpy as np
import mujoco

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DESC = os.path.join(ROOT, "robot_descriptions")

# ---- table design (placeholders, we tune these after workspace analysis) ----
TABLE_H = 0.25   # table top height (m)
TABLE_L = 0.60   # length along x (m)
TABLE_W = 0.80   # width along y (m)
TABLE_X = 0.50   # table centre x (m)
CUBE_HALF = 0.02 # 4 cm cube
TRAY_POS = (0.45, -0.25)  # tray centre (x, y)


def build_scene(cube_xy=(0.45, 0.15), cube_yaw=0.0):
    arm = mujoco.MjSpec.from_file(
        os.path.join(DESC, "single_arm_heal_effort_actuation_rs_mj.xml"))
    grip = mujoco.MjSpec.from_file(
        os.path.join(DESC, "robotiq_2f85_v4", "2f85.xml"))
    arm.attach(grip, prefix="gripper/", site=arm.site("right_center"))

    box = mujoco.mjtGeom.mjGEOM_BOX
    wb = arm.worldbody

    # table: top surface at z = TABLE_H
    wb.add_geom(name="table", type=box,
                size=[TABLE_L / 2, TABLE_W / 2, TABLE_H / 2],
                pos=[TABLE_X, 0, TABLE_H / 2],
                rgba=[0.6, 0.45, 0.3, 1], friction=[1, 0.005, 0.0001])

    # cube with free joint
    cube = wb.add_body(name="cube",
                       pos=[cube_xy[0], cube_xy[1], TABLE_H + CUBE_HALF + 0.001],
                       quat=[np.cos(cube_yaw / 2), 0, 0, np.sin(cube_yaw / 2)])
    cube.add_freejoint()
    cube.add_geom(name="cube_geom", type=box,
                  size=[CUBE_HALF] * 3, rgba=[1, 0, 0, 1],
                  friction=[1.5, 0.05, 0.001], mass=0.05)

    # tray: flat base + 4 low walls
    tx, ty = TRAY_POS
    hb, wall_h, t = 0.08, 0.03, 0.005
    tray = wb.add_body(name="tray", pos=[tx, ty, TABLE_H])
    tray.add_geom(name="tray_base", type=box, size=[hb, hb, t],
                  pos=[0, 0, t], rgba=[1, 0.9, 0.1, 1])
    for n, p, s in [("w_px", [hb, 0, wall_h], [t, hb, wall_h]),
                    ("w_nx", [-hb, 0, wall_h], [t, hb, wall_h]),
                    ("w_py", [0, hb, wall_h], [hb, t, wall_h]),
                    ("w_ny", [0, -hb, wall_h], [hb, t, wall_h])]:
        tray.add_geom(name="tray_" + n, type=box, size=s, pos=p,
                      rgba=[1, 0.9, 0.1, 1])

    return arm.compile()


if __name__ == "__main__":
    import mujoco.viewer
    model = build_scene()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print("Joints:", [model.joint(i).name for i in range(model.njnt)])
    print("Actuators:", [model.actuator(i).name for i in range(model.nu)])

    with mujoco.viewer.launch_passive(model, data) as v:
        while v.is_running():
            for a in range(model.nu):
                dof = model.actuator_trnid[a][0]
                if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
                    data.ctrl[a] = data.qfrc_bias[model.jnt_dofadr[dof]]
            mujoco.mj_step(model, data)
            v.sync()
