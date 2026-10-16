import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer

# Define XML scene string with ground plane, table, tray, and cube
SCENE_XML = """
<mujoco model="basic_scene_with_tray">
  <option timestep="0.002"/>

  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.8 0.8 0.8"/>
  </visual>

  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.1 0.2 0.3" rgb2="0.2 0.3 0.4"/>
    <material name="grid_mat" texture="grid" texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <!-- Lighting -->
    <light pos="0 0 3" dir="0 0 -1" castshadow="true"/>

    <!-- Ground Plane -->
    <geom name="floor" type="plane" size="5 5 0.1" material="grid_mat"/>

    <!-- Table -->
    <body name="table" pos="0.5 0 0.2">
      <geom type="box" size="0.25 0.35 0.2" rgba="0.6 0.4 0.2 1"/>
    </body>

    <!-- Tray Target Zone (Placed on the left side of table) -->
    <body name="tray" pos="0.5 -0.18 0.405">
      <!-- Tray Floor -->
      <geom type="box" size="0.08 0.08 0.005" rgba="0.2 0.2 0.2 1"/>
      <!-- Tray Borders -->
      <geom type="box" size="0.08 0.005 0.02" pos="0 0.08 0.01" rgba="0.1 0.1 0.1 1"/>
      <geom type="box" size="0.08 0.005 0.02" pos="0 -0.08 0.01" rgba="0.1 0.1 0.1 1"/>
      <geom type="box" size="0.005 0.08 0.02" pos="0.08 0 0.01" rgba="0.1 0.1 0.1 1"/>
      <geom type="box" size="0.005 0.08 0.02" pos="-0.08 0 0.01" rgba="0.1 0.1 0.1 1"/>
    </body>

    <!-- Target Cube (Placed on the right side of table) -->
    <body name="cube" pos="0.5 0.15 0.425">
      <joint type="free"/>
      <geom type="box" size="0.02 0.02 0.02" rgba="0.9 0.1 0.1 1" mass="0.05"/>
    </body>
  </worldbody>
</mujoco>
"""

def main():
    print("Loading environment with Table, Tray, and Red Cube...")
    
    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    data = mujoco.MjData(model)

    print("Launching viewer. Press ESC or close window to exit.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

if __name__ == "__main__":
    main()
