import numpy as np
import random
import colorsys

def get_contrasting_colors():
    hue_sky = random.random()
    hue_floor = (hue_sky + random.uniform(0.33, 0.66)) % 1.0
    hue_object = (hue_floor + random.uniform(0.33, 0.66)) % 1.0

    r_sky, g_sky, b_sky = colorsys.hsv_to_rgb(hue_sky, random.uniform(0.1, 0.3), random.uniform(0.6, 0.9))
    r_flr, g_flr, b_flr = colorsys.hsv_to_rgb(hue_floor, random.uniform(0.15, 0.35), random.uniform(0.2, 0.45))
    r_obj, g_obj, b_obj = colorsys.hsv_to_rgb(hue_object, random.uniform(0.7, 1.0), random.uniform(0.6, 0.9))

    return (f"{r_sky:.3f} {g_sky:.3f} {b_sky:.3f}",
            f"{r_flr:.3f} {g_flr:.3f} {b_flr:.3f}",
            f"{r_obj:.3f} {g_obj:.3f} {b_obj:.3f}")


def generate_beveled_box_mesh(size_x, size_y, size_z, radius, resolution=8):
    hx = max(0.001, size_x - radius)
    hy = max(0.001, size_y - radius)
    hz = max(0.001, size_z - radius)

    sphere_points = []
    for i in range(resolution + 1):
        lat = (np.pi / 2) * (i / resolution)
        for j in range(resolution + 1):
            lon = (np.pi / 2) * (j / resolution)
            x = np.cos(lat) * np.cos(lon)
            y = np.cos(lat) * np.sin(lon)
            z = np.sin(lat)
            sphere_points.append((x, y, z))

    vertices = []
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                for px, py, pz in sphere_points:
                    vx = sx * hx + sx * radius * px
                    vy = sy * hy + sy * radius * py
                    vz = sz * hz + sz * radius * pz
                    vertices.append(f"{vx:.4f} {vy:.4f} {vz:.4f}")

    return " ".join(vertices)


def generate_noise_based_com(shape_type, size_x, size_y, size_z):
    # The Mass Distribution Fix: Uniformly spawn anywhere from 10% to 90% of the distance to the edge
    # This samples uniformly across the radius (distance), removing extreme edge volumetric bias.
    extremity = np.random.uniform(0.1, 0.9)

    if shape_type in ["box", "beveled_box"]:
        point = np.random.uniform(-extremity, extremity, 3)
        face = random.randint(0, 2)
        point[face] = extremity * random.choice([-1, 1])
        noise_x, noise_y, noise_z = point

    elif shape_type == "cylinder":
        if random.random() < 0.5:
            theta = np.random.uniform(0, 2 * np.pi)
            r = extremity * np.sqrt(random.random())
            noise_x = r * np.cos(theta)
            noise_y = r * np.sin(theta)
            noise_z = extremity * random.choice([-1, 1])
        else:
            theta = np.random.uniform(0, 2 * np.pi)
            noise_x = extremity * np.cos(theta)
            noise_y = extremity * np.sin(theta)
            noise_z = np.random.uniform(-extremity, extremity)

    elif shape_type == "ellipsoid":
        u = np.random.uniform(0, 1)
        v = np.random.uniform(0, 1)
        theta = 2 * np.pi * u
        phi = np.arccos(2 * v - 1)
        noise_x = extremity * np.sin(phi) * np.cos(theta)
        noise_y = extremity * np.sin(phi) * np.sin(theta)
        noise_z = extremity * np.cos(phi)

    return noise_x * size_x, noise_y * size_y, noise_z * size_z


def calculate_exact_inertia(shape_type, mass, size_x, size_y, size_z):
    if shape_type in ["box", "beveled_box"]:
        ix = (1 / 3) * mass * (size_y ** 2 + size_z ** 2)
        iy = (1 / 3) * mass * (size_x ** 2 + size_z ** 2)
        iz = (1 / 3) * mass * (size_x ** 2 + size_y ** 2)
    elif shape_type == "cylinder":
        ix = iy = (1 / 12) * mass * (3 * size_x ** 2 + 4 * size_z ** 2)
        iz = (1 / 2) * mass * size_x ** 2
    elif shape_type == "ellipsoid":
        ix = (1 / 5) * mass * (size_y ** 2 + size_z ** 2)
        iy = (1 / 5) * mass * (size_x ** 2 + size_z ** 2)
        iz = (1 / 5) * mass * (size_x ** 2 + size_y ** 2)

    return f'diaginertia="{ix:.6f} {iy:.6f} {iz:.6f}"'


def generate_randomized_xml(cam_mode="STABLE"):
    gravity_z = np.random.uniform(-9.834, -9.764)
    air_density = np.random.uniform(1.1, 1.3)

    color_sky_rgb, color_floor_rgb, color_obj_rgb = get_contrasting_colors()

    is_close_target = random.random() < 0.15 and cam_mode != "STATIC"

    # The Optics: Locked Camera FOV randomization strictly between [65, 85] degrees perfectly mimicking phone lenses
    camera_fovy = np.random.uniform(65, 85)

    # The Scale Fix (Camera Distance): Randomize the physical distance of the camera from the object (0.5m to 4.0m)
    radius = np.random.uniform(0.5, 4.0)

    if cam_mode == "STATIC":
        drop_height = np.random.uniform(2.0, 5.0)
    elif is_close_target:
        drop_height = np.random.uniform(1.0, 2.5)
    else:
        drop_height = np.random.uniform(2.5, 4.5)

    azimuth = np.random.uniform(0, 2 * np.pi)
    elevation = np.random.uniform(np.deg2rad(15), np.deg2rad(35))

    camera_x = radius * np.cos(elevation) * np.cos(azimuth)
    camera_y = radius * np.cos(elevation) * np.sin(azimuth)
    camera_z = radius * np.sin(elevation) + (drop_height * 0.4)

    floor_texture = random.choices(["checker", "gradient", "flat"], weights=[0.2, 0.4, 0.4])[0]

    # The Textures: Heavy bias towards high-contrast macro-textures (like ArUco checkerboards) to prevent visual aliasing
    object_texture = random.choices(["checker", "gradient"], weights=[0.8, 0.2])[0]
    texrepeat_str = f"{random.choice([2, 3, 4])} {random.choice([2, 3, 4])}"

    # Generate a stark contrasting secondary color based on luminance
    rgb1_parts = list(map(float, color_obj_rgb.split()))
    luminance = 0.299 * rgb1_parts[0] + 0.587 * rgb1_parts[1] + 0.114 * rgb1_parts[2]
    rgb2_obj = "0.05 0.05 0.05" if luminance > 0.5 else "0.95 0.95 0.95"

    def random_gray():
        val = np.random.uniform(0.2, 0.8)
        return f"{val:.3f} {val:.3f} {val:.3f}"

    fric_str = f"{np.random.uniform(0.3, 0.7):.3f} {np.random.uniform(0.01, 0.05):.4f} {np.random.uniform(0.001, 0.005):.4f}"

    damping_ratio = np.random.uniform(0.3, 0.7)
    soft_solref = f"0.015 {damping_ratio:.3f}"
    soft_solimp = "0.9 0.95 0.001"

    has_ramp = random.random() < 0.4
    ramp_euler = f"0 {np.random.uniform(10, 25):.1f} 0"
    ramp_geometry = f'<geom name="ramp" type="box" size="1.5 1.5 0.1" pos="0 0 0.0" euler="{ramp_euler}" material="mat_ramp" solref="{soft_solref}" solimp="{soft_solimp}" friction="{fric_str}" margin="0.002"/>' if has_ramp else ""

    distractors_xml = ""
    for _ in range(random.randint(1, 4)):
        dx, dy = np.random.uniform(-1.2, 1.2, 2)
        if abs(dx) < 0.4 and abs(dy) < 0.4: continue
        dtype = random.choice(["box", "cylinder", "sphere"])
        if dtype == "box":
            size_str = "0.06 0.06 0.06"
        elif dtype == "cylinder":
            size_str = "0.06 0.06"
        else:  # sphere
            size_str = "0.06"

        distractors_xml += f"""
        <body pos="{dx:.3f} {dy:.3f} {np.random.uniform(1.0, 2.0):.3f}">
            <freejoint/>
            <geom type="{dtype}" size="{size_str}" rgba="{random_gray()} 1" material="mat_bg" solref="{soft_solref}" solimp="{soft_solimp}" friction="{fric_str}" margin="0.002"/>
        </body>
        """

    shape_type = random.choices(["box", "beveled_box", "cylinder", "ellipsoid"], weights=[0.35, 0.35, 0.15, 0.15])[0]

    if shape_type == "cylinder":
        if is_close_target:
            size_x = size_y = np.random.uniform(0.025, 0.05)
            size_z = np.random.uniform(0.025, 0.06)
        else:
            size_x = size_y = np.random.uniform(0.12, 0.22)
            size_z = np.random.uniform(0.12, 0.22)
        size_string = f"{size_x:.3f} {size_z:.3f}"
    else:
        if is_close_target:
            size_x = np.random.uniform(0.025, 0.05)
            size_y = np.random.uniform(0.025, 0.05)
            size_z = np.random.uniform(0.025, 0.05)
        else:
            size_x = np.random.uniform(0.19, 0.32)
            size_y = np.random.uniform(0.12, 0.22)
            size_z = np.random.uniform(0.09, 0.17)
        size_string = f"{size_x:.3f} {size_y:.3f} {size_z:.3f}"

    if shape_type == "beveled_box":
        bevel_radius = np.random.uniform(0.015, 0.04)
        mesh_str = generate_beveled_box_mesh(size_x, size_y, size_z, bevel_radius, resolution=8)
        mesh_asset = f'<mesh name="beveled_box_mesh" vertex="{mesh_str}"/>'
        geom_attributes = f'type="mesh" mesh="beveled_box_mesh"'
    else:
        mesh_asset = ""
        geom_attributes = f'type="{shape_type}" size="{size_string}"'

    com_x, com_y, com_z = generate_noise_based_com(shape_type, size_x, size_y, size_z)
    mass = np.random.uniform(1.5, 4.0)
    inertia_string = calculate_exact_inertia(shape_type, mass, size_x, size_y, size_z)

    com_magnitude = np.sqrt(com_x ** 2 + com_y ** 2 + com_z ** 2)

    xml_string = f"""
    <mujoco model="statera_poc">
        <compiler angle="degree" coordinate="local" balanceinertia="true"/>
        <option gravity="0 0 {gravity_z:.4f}" timestep="0.001" solver="Newton" iterations="150" tolerance="1e-10" cone="elliptic" jacobian="dense" density="{air_density:.3f}" viscosity="1.8e-5"/>

        <visual>
            <global fovy="{camera_fovy:.2f}" offwidth="640" offheight="640"/>
            <quality shadowsize="4096" offsamples="8"/>
        </visual>
        <asset>
            {mesh_asset}
            <texture type="skybox" builtin="gradient" rgb1="{color_sky_rgb}" rgb2="0.05 0.05 0.05" width="512" height="512"/>

            <texture name="tex_floor" type="2d" builtin="{floor_texture}" rgb1="{color_floor_rgb}" rgb2="{random_gray()}" width="512" height="512"/>
            <texture name="tex_ramp" type="2d" builtin="flat" rgb1="{color_floor_rgb}" rgb2="{random_gray()}" width="512" height="512"/>

            <texture name="tex_obj" type="2d" builtin="{object_texture}" rgb1="{color_obj_rgb}" rgb2="{rgb2_obj}" width="512" height="512"/>

            <material name="mat_floor" texture="tex_floor" texrepeat="5 5" specular="0.2" shininess="0.1"/>
            <material name="mat_ramp" texture="tex_ramp" specular="0.1" shininess="0.1"/>
            <material name="mat_bg" specular="0.2" shininess="0.2"/>
            <material name="mat_obj" texture="tex_obj" texrepeat="{texrepeat_str}" texuniform="true" specular="{np.random.uniform(0.3, 0.8):.2f}" shininess="{np.random.uniform(0.5, 0.9):.2f}"/>
        </asset>

        <worldbody>
            <body name="camera_tracker" pos="0 0 1.0" mocap="true">
                <geom type="sphere" size="0.01" rgba="0 0 0 0" contype="0" conaffinity="0"/>
            </body>
            <camera name="main_cam" pos="{camera_x:.3f} {camera_y:.3f} {camera_z:.3f}" mode="targetbody" target="camera_tracker"/>

            <light pos="{np.random.uniform(1.0, 3.0):.3f} {np.random.uniform(1.0, 3.0):.3f} {np.random.uniform(2.5, 4.0):.3f}" 
                   dir="-2 -2 -3" diffuse="{np.random.uniform(0.6, 1.0):.2f} {np.random.uniform(0.6, 1.0):.2f} {np.random.uniform(0.6, 1.0):.2f}" 
                   specular="{np.random.uniform(0.2, 0.5):.2f} {np.random.uniform(0.2, 0.5):.2f} {np.random.uniform(0.2, 0.5):.2f}" castshadow="true"/>
            <light pos="-2 -2 2" dir="2 2 -2" diffuse="{np.random.uniform(0.2, 0.5):.2f} {np.random.uniform(0.2, 0.5):.2f} {np.random.uniform(0.2, 0.5):.2f}" castshadow="false"/>

            <geom name="floor" type="plane" size="15 15 0.1" material="mat_floor" friction="{fric_str}" solref="{soft_solref}" solimp="{soft_solimp}"/>

            {ramp_geometry}
            {distractors_xml}

            <body name="target_object" pos="0 0 {drop_height:.3f}">
                <freejoint/>
                <inertial pos="{com_x:.4f} {com_y:.4f} {com_z:.4f}" mass="{mass:.3f}" {inertia_string}/>
                <geom name="shell" {geom_attributes} material="mat_obj" friction="{fric_str}" density="0" solref="{soft_solref}" solimp="{soft_solimp}" margin="0.002"/>
            </body>
        </worldbody>
    </mujoco>
    """
    return xml_string, com_magnitude