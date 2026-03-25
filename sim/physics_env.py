import mujoco
import numpy as np
import h5py
import albumentations as A
from generator import generate_randomized_xml

# Only applying Gaussian noise (Film Grain). Lens distortion is now handled natively in MuJoCo!
frame_noise = A.Compose([
    A.GaussNoise(std_range=(0.01, 0.04), p=0.8),
])


def apply_episode_white_balance(frames):
    contrast = np.random.uniform(0.8, 1.2)
    brightness = np.random.uniform(-30, 30)
    r_shift, g_shift, b_shift = np.random.uniform(-20, 20, 3)
    processed = []
    for f in frames:
        f = f.astype(np.float32) * contrast + brightness
        f[:, :, 0] += r_shift
        f[:, :, 1] += g_shift
        f[:, :, 2] += b_shift
        processed.append(np.clip(f, 0, 255).astype(np.uint8))
    return processed


def project_3d_to_2d(model, data, cam_name="main_cam", resolution=224):
    """Guaranteed mathematical perfection. No sub-pixel drifting."""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    cam_pos = data.cam_xpos[cam_id]
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3)
    fovy = model.cam_fovy[cam_id]

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
    com_world = data.xipos[body_id]

    pt_cam = cam_mat.T @ (com_world - cam_pos)
    focal_length = 0.5 * resolution / np.tan(np.deg2rad(fovy) / 2)

    u = (pt_cam[0] / -pt_cam[2]) * focal_length + (resolution / 2)
    v = (-pt_cam[1] / -pt_cam[2]) * focal_length + (resolution / 2)
    return float(u), float(v), float(-pt_cam[2])


def generate_statera_episode(hdf5_file, episode_idx):
    xml_str = generate_randomized_xml()
    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=224, width=224)

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
    tracker_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "camera_tracker")
    mocap_id = model.body_mocapid[tracker_id]
    dof_start = model.body_dofadr[body_id]

    data.qvel[dof_start: dof_start + 3] = np.random.uniform(-0.5, 0.5, 3)
    data.qvel[dof_start + 3: dof_start + 6] = np.random.uniform(-15, 15, 3)

    mujoco.mj_forward(model, data)
    data.mocap_pos[mocap_id] = data.xipos[body_id].copy()

    frames, labels, z_heights = [], [], []
    steady_wind = np.random.uniform(-0.5, 0.5, 3)

    for step in range(120):
        data.xfrc_applied[body_id][:3] = steady_wind

        t = data.time
        obj_pos = data.xipos[body_id]
        current_cam_target = data.mocap_pos[mocap_id]

        pan_speed = 0.15
        new_target = current_cam_target + pan_speed * (obj_pos - current_cam_target)

        jitter_x = 0.015 * np.sin(10 * t) + 0.005 * np.cos(23 * t)
        jitter_y = 0.015 * np.cos(12 * t) + 0.005 * np.sin(27 * t)
        jitter_z = 0.010 * np.sin(15 * t)
        data.mocap_pos[mocap_id] = new_target + np.array([jitter_x, jitter_y, jitter_z])

        for _ in range(5):
            mujoco.mj_step(model, data)

        renderer.update_scene(data, camera="main_cam")

        frames.append(renderer.render())
        u, v, z = project_3d_to_2d(model, data)
        labels.append([u, v, z])
        z_heights.append(data.xipos[body_id][2])

    impact_frame = np.argmin(z_heights)
    best_start = max(0, min(impact_frame - 6, len(frames) - 16))

    best_frames = frames[best_start: best_start + 16]
    best_labels = labels[best_start: best_start + 16]

    wb_frames = apply_episode_white_balance(best_frames)

    noisy_frames = []
    for frame in wb_frames:
        noisy_frames.append(frame_noise(image=frame)['image'])

    hdf5_file.create_dataset(f"video_{episode_idx}", data=np.array(noisy_frames, dtype=np.uint8), compression="lzf")
    hdf5_file.create_dataset(f"label_{episode_idx}", data=np.array(best_labels, dtype=np.float32))

    print(f"Generated Episode {episode_idx} | Impact Frame Captured: {impact_frame}")


if __name__ == "__main__":
    print("Initializing STATERA Pipeline (V3 - Mathematical Precision)...")
    with h5py.File("statera_poc.hdf5", "w") as f:
        for i in range(5):
            generate_statera_episode(f, i)
    print("V3 Dataset Generation Complete.")