import time
import math

import mujoco
import mujoco.viewer
import numpy as np
import glfw


file_path = "./equality.xml"
m = mujoco.MjModel.from_xml_path(file_path)
d = mujoco.MjData(m)

# Set initial velocity to shoot the box complex towards the wall and rotate it
body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'box000')
joint_id = m.body_jntadr[body_id]
qvel_adr = m.jnt_dofadr[joint_id]
d.qvel[qvel_adr : qvel_adr + 6] = [80.0, 0.0, 0.0, 0.0, 10.0, 0.0]

# Get sensor id for wall force
sensor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, 'wall_force')

eq_active = True
reset_active = False

def key_callback(key: int):
    global eq_active,reset_active
    if key == glfw.KEY_SPACE:
        eq_active = not eq_active
    elif key == glfw.KEY_BACKSPACE:
        reset_active = True

with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
    while viewer.is_running():
        step_start = time.time()

        mujoco.mj_step(m, d)
        wall_force = d.sensordata[sensor_id : sensor_id + 3]  # Get force on wall
        
        if not eq_active and np.linalg.norm(wall_force) > 0.01:
            d.eq_active[:] = 0

        viewer.sync()
        if reset_active:
            d.qvel[qvel_adr : qvel_adr + 6] = [80.0, 0.0, 0.0, 0.0, 10.0, 0.0]
            reset_active = False
            
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
