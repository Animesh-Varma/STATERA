import numpy as np
import random
import colorsys


def get_contrasting_colors():
    """Generates a background and object color that are mathematically guaranteed to contrast."""
    # 1. Background: Muted, realistic room colors (lower saturation)
    h_bg = random.random()
    s_bg = random.uniform(0.1, 0.4)
    v_bg = random.uniform(0.4, 0.7)

    # 2. Object: High saturation, shifted hue (opposite side of the color wheel)
    h_obj = (h_bg + random.uniform(0.3, 0.7)) % 1.0
    s_obj = random.uniform(0.7, 1.0)
    v_obj = random.uniform(0.6, 0.9)

    r_bg, g_bg, b_bg = colorsys.hsv_to_rgb(h_bg, s_bg, v_bg)
    r_obj, g_obj, b_obj = colorsys.hsv_to_rgb(h_obj, s_obj, v_obj)

    return f"{r_bg:.3f} {g_bg:.3f} {b_bg:.3f}", f"{r_obj:.3f} {g_obj:.3f} {b_obj:.3f}"


def generate_randomized_xml():
    # 1. Realistic Earth-bound Physics
    gravity_z = np.random.uniform(-10.2, -9.5)  # Close to Earth's -9.81

    # Anti-Camouflage Colors
    bg_rgb, obj_rgb = get_contrasting_colors()

    # 2. Realistic Studio Lighting (Key + Fill Light)
    key_x, key_y = np.random.uniform(-2, 2, 2)
    # Fill light is opposite to the key light to soften shadows
    fill_x, fill_y = -key_x, -key_y

    # 3. Camera Orbit Math
    radius = np.random.uniform(1.8, 2.5)
    azimuth = np.random.uniform(0, 2 * np.pi)
    elevation = np.random.uniform(np.deg2rad(25), np.deg2rad(55))
    cam_x = radius * np.cos(elevation) * np.cos(azimuth)
    cam_y = radius * np.cos(elevation) * np.sin(azimuth)
    cam_z = radius * np.sin(elevation) + 0.5

    # 4. Textures & Specular Materials (Shininess)
    tex_types = ["checker", "gradient", "flat"]
    floor_tex = random.choice(tex_types)
    ramp_tex = random.choice(tex_types)

    def rand_gray():
        return f"{np.random.uniform(0.2, 0.8)} {np.random.uniform(0.2, 0.8)} {np.random.uniform(0.2, 0.8)}"

    # 5. Environment generation
    has_ramp = random.random() > 0.3
    ramp_geom = f'<geom name="ramp" type="box" size="1.5 1.5 0.1" pos="0 0 0.1" euler="0 {np.random.uniform(10, 25)} 0" material="mat_ramp"/>' if has_ramp else ""

    distractors = ""
    for _ in range(random.randint(1, 3)):
        dx, dy = np.random.uniform(-1.5, 1.5, 2)
        if abs(dx) < 0.6 and abs(dy) < 0.6: continue
        dtype = random.choice(["box", "sphere", "cylinder"])
        distractors += f'<geom type="{dtype}" size="0.1 0.1 0.1" pos="{dx} {dy} 0.1" rgba="{rand_gray()} 1" material="mat_bg"/>\n'

    # 6. Target Object Generator
    shapes = ["box", "sphere", "cylinder", "ellipsoid"]
    shape = random.choice(shapes)
    size_str = f"{np.random.uniform(0.06, 0.15)} {np.random.uniform(0.06, 0.15)} {np.random.uniform(0.06, 0.15)}"
    if shape == "sphere":
        size_str = f"{np.random.uniform(0.08, 0.12)}"
    elif shape == "cylinder":
        size_str = f"{np.random.uniform(0.05, 0.12)} {np.random.uniform(0.05, 0.12)}"

    # The Hidden CoM Offset
    com_x, com_y, com_z = np.random.uniform(-0.03, 0.03, 3)

    xml_string = f"""
    <mujoco model="statera_v3_realistic">
        <compiler angle="degree" coordinate="local"/>
        <option gravity="0 0 {gravity_z}" timestep="0.01"/>

        <visual>
            <map fogstart="1.5" fogend="6.0"/>
            <global fovy="45"/>
        </visual>

        <asset>
            <texture type="skybox" builtin="gradient" rgb1="{bg_rgb}" rgb2="0.1 0.1 0.1" width="512" height="512"/>
            <texture name="tex_floor" type="2d" builtin="{floor_tex}" rgb1="{bg_rgb}" rgb2="{rand_gray()}" width="512" height="512" mark="random" markrgb="1 1 1"/>
            <texture name="tex_ramp" type="2d" builtin="{ramp_tex}" rgb1="{bg_rgb}" rgb2="{rand_gray()}" width="512" height="512"/>

            <!-- Added specular shininess for realism -->
            <material name="mat_floor" texture="tex_floor" texrepeat="5 5" specular="0.2" shininess="0.1"/>
            <material name="mat_ramp" texture="tex_ramp" texrepeat="2 2" specular="0.1" shininess="0.1"/>
            <material name="mat_bg" specular="0.3" shininess="0.5"/>
            <material name="mat_obj" rgba="{obj_rgb} 1" specular="0.5" shininess="0.8"/> <!-- High specular for the target object -->
        </asset>

        <worldbody>
            <body name="look_target" pos="0 0 0.4"/>
            <camera name="main_cam" pos="{cam_x} {cam_y} {cam_z}" mode="targetbody" target="look_target"/>

            <!-- Studio Lighting: Key Light (Warm/Bright) + Fill Light (Cool/Dim) -->
            <light pos="{key_x} {key_y} 3" dir="{-key_x} {-key_y} -3" ambient="0.1 0.1 0.1" diffuse="0.7 0.7 0.6" specular="0.3 0.3 0.3" castshadow="true"/>
            <light pos="{fill_x} {fill_y} 2" dir="{-fill_x} {-fill_y} -2" diffuse="0.2 0.2 0.3" specular="0.1 0.1 0.1" castshadow="false"/>

            <!-- Floor -->
            <geom name="floor" type="plane" size="4 4 0.1" material="mat_floor" friction="{np.random.uniform(0.2, 0.8)}"/>

            {ramp_geom}
            {distractors}

            <!-- The Tumbling Target Object -->
            <body name="target_object" pos="0 0 1.5">
                <freejoint/>
                <inertial pos="{com_x} {com_y} {com_z}" mass="{np.random.uniform(0.8, 2.0)}" diaginertia="0.01 0.01 0.01"/>
                <!-- Object uses the contrasting color material -->
                <geom name="shell" type="{shape}" size="{size_str}" material="mat_obj" friction="{np.random.uniform(0.2, 0.8)}" density="0"/>
            </body>
        </worldbody>
    </mujoco>
    """
    return xml_string