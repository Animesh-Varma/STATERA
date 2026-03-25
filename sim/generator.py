import numpy as np
import random
import colorsys


def get_contrasting_colors():
    h_bg = random.random()
    s_bg = random.uniform(0.1, 0.3)
    v_bg = random.uniform(0.4, 0.7)
    h_obj = (h_bg + random.uniform(0.3, 0.7)) % 1.0
    r_bg, g_bg, b_bg = colorsys.hsv_to_rgb(h_bg, s_bg, v_bg)
    r_obj, g_obj, b_obj = colorsys.hsv_to_rgb(h_obj, random.uniform(0.7, 1.0), random.uniform(0.6, 0.9))
    return f"{r_bg:.3f} {g_bg:.3f} {b_bg:.3f}", f"{r_obj:.3f} {g_obj:.3f} {b_obj:.3f}"


def generate_randomized_xml():
    gravity_z = np.random.uniform(-10.2, -9.5)
    bg_rgb, obj_rgb = get_contrasting_colors()

    radius = np.random.uniform(1.2, 1.8)
    azimuth = np.random.uniform(0, 2 * np.pi)
    elevation = np.random.uniform(np.deg2rad(15), np.deg2rad(60))
    cam_x = radius * np.cos(elevation) * np.cos(azimuth)
    cam_y = radius * np.cos(elevation) * np.sin(azimuth)
    cam_z = radius * np.sin(elevation) + 0.2

    floor_tex = random.choices(["checker", "gradient", "flat"], weights=[0.1, 0.4, 0.5])[0]
    # FIX: Increased chance of checkerboard on the object specifically
    obj_tex = random.choices(["checker", "gradient", "flat"], weights=[0.35, 0.3, 0.35])[0]

    def rand_gray():
        return f"{np.random.uniform(0.2, 0.8)} {np.random.uniform(0.2, 0.8)} {np.random.uniform(0.2, 0.8)}"

    has_ramp = random.random() > 0.3
    ramp_geom = f'<geom name="ramp" type="box" size="1.5 1.5 0.1" pos="0 0 0.1" euler="0 {np.random.uniform(10, 25)} 0" material="mat_ramp"/>' if has_ramp else ""

    # FIX: Brought distractors back!
    distractors = ""
    for _ in range(random.randint(1, 4)):
        dx, dy = np.random.uniform(-1.2, 1.2, 2)
        if abs(dx) < 0.4 and abs(dy) < 0.4: continue
        dtype = random.choice(["box", "cylinder", "sphere"])
        distractors += f'<geom type="{dtype}" size="0.06 0.06 0.06" pos="{dx} {dy} 0.06" rgba="{rand_gray()} 1" material="mat_bg"/>\n'

    shape = random.choices(["box", "cylinder", "ellipsoid"], weights=[0.8, 0.1, 0.1])[0]

    sx, sy, sz = np.random.uniform(0.08, 0.15), np.random.uniform(0.04, 0.08), np.random.uniform(0.04, 0.08)
    if shape == "cylinder":
        sx, sy, sz = np.random.uniform(0.04, 0.08), sx, sx
    elif shape == "ellipsoid":
        sx, sy, sz = np.random.uniform(0.08, 0.12), np.random.uniform(0.05, 0.08), np.random.uniform(0.05, 0.08)

    size_str = f"{sx} {sz}" if shape == "cylinder" else f"{sx} {sy} {sz}"

    if random.random() < 0.15:
        com_x = com_y = com_z = 0.0
    else:
        bound_x = sx
        bound_y = sx if shape == "cylinder" else sy
        bound_z = sz
        com_x = np.random.uniform(-bound_x * 0.75, bound_x * 0.75)
        com_y = np.random.uniform(-bound_y * 0.75, bound_y * 0.75)
        com_z = np.random.uniform(-bound_z * 0.75, bound_z * 0.75)

    # FIX: Random FOV simulates everything from Telephoto (30) to GoPro Wide-Angle (90)
    camera_fovy = np.random.uniform(30, 90)

    # FIX: texrepeat and texuniform map checkerboards to the individual faces of boxes!
    obj_texrepeat = f"{random.randint(1, 4)} {random.randint(1, 4)}"

    xml_string = f"""
    <mujoco model="statera_v3">
        <compiler angle="degree" coordinate="local"/>
        <option gravity="0 0 {gravity_z}" timestep="0.002" solver="Newton" iterations="50" tolerance="1e-10"/>

        <visual>
            <global fovy="{camera_fovy}"/>
            <quality shadowsize="4096" offsamples="4"/>
        </visual>

        <asset>
            <texture type="skybox" builtin="gradient" rgb1="{bg_rgb}" rgb2="0.1 0.1 0.1" width="512" height="512"/>
            <texture name="tex_floor" type="2d" builtin="{floor_tex}" rgb1="{bg_rgb}" rgb2="{rand_gray()}" width="512" height="512"/>
            <texture name="tex_ramp" type="2d" builtin="flat" rgb1="{bg_rgb}" rgb2="{rand_gray()}" width="512" height="512"/>
            <texture name="tex_obj" type="2d" builtin="{obj_tex}" rgb1="{obj_rgb}" rgb2="{rand_gray()}" width="512" height="512"/>

            <material name="mat_floor" texture="tex_floor" texrepeat="5 5" specular="0.2" shininess="0.1"/>
            <material name="mat_ramp" texture="tex_ramp" specular="0.1" shininess="0.1"/>
            <material name="mat_bg" specular="0.2" shininess="0.2"/>
            <!-- texuniform guarantees clean box face mapping -->
            <material name="mat_obj" texture="tex_obj" texrepeat="{obj_texrepeat}" texuniform="true" specular="0.6" shininess="0.8"/>
        </asset>

        <worldbody>
            <body name="camera_tracker" pos="0 0 1.0" mocap="true">
                <geom type="sphere" size="0.01" rgba="0 0 0 0" contype="0" conaffinity="0"/>
            </body>
            <camera name="main_cam" pos="{cam_x} {cam_y} {cam_z}" mode="targetbody" target="camera_tracker"/>

            <light pos="2 2 3" dir="-2 -2 -3" diffuse="0.8 0.8 0.7" specular="0.3 0.3 0.3" castshadow="true"/>
            <light pos="-2 -2 2" dir="2 2 -2" diffuse="0.3 0.3 0.4" castshadow="false"/>

            <geom name="floor" type="plane" size="4 4 0.1" material="mat_floor" friction="{np.random.uniform(0.3, 0.8)}"/>

            {ramp_geom}
            {distractors}

            <body name="target_object" pos="0 0 1.2">
                <freejoint/>
                <inertial pos="{com_x} {com_y} {com_z}" mass="{np.random.uniform(0.8, 2.0)}" diaginertia="0.01 0.01 0.01"/>
                <geom name="shell" type="{shape}" size="{size_str}" material="mat_obj" friction="{np.random.uniform(0.3, 0.8)}" density="0"/>
            </body>
        </worldbody>
    </mujoco>
    """
    return xml_string