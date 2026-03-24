import mujoco
import numpy as np
import h5py
import cv2
import albumentations as A
from generator import generate_randomized_xml

# 1. Subtle Frame-Level Noise (Film Grain / Mild Lens warp applied per-frame)
frame_noise = A.Compose([
    A.GaussNoise(std_range=(0.01, 0.04), p=0.5),
    A.OpticalDistortion(distort_limit=0.03, p=0.4)  # Very subtle lens warp
])


def apply_episode_white_balance(frames):
    """
    Applies White Balance, Contrast, and Brightness shifts ONCE per episode.
    This guarantees the lighting stays stable across the 16 frames,
    but varies wildly between different videos.
    """
    contrast = np.random.uniform(0.8, 1.2)
    brightness = np.random.uniform(-30, 30)

    # Random RGB Tint (Simulating Tungsten/Fluorescent/Daylight)
    r_shift = np.random.uniform(-20, 20)
    g_shift = np.random.uniform(-20, 20)
    b_shift = np.random.uniform(-20, 20)

    processed = []
    for f in frames:
        f = f.astype(np.float32)
        f = f * contrast + brightness
        f[:, :, 0] += r_shift  # MuJoCo renders RGB
        f[:, :, 1] += g_shift
        f[:, :, 2] += b_shift
        f = np.clip(f, 0, 255).astype(np.uint8)
        processed.append(f)
    return processed


def project_3d_to_2d(model, data, cam_name="main_cam", resolution=224):
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
    z_depth = -pt_cam[2]

    return float(u), float(v), float(z_depth)


def generate_statera_episode(hdf5_file, episode_idx):
    xml_str = generate_randomized_xml()
    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=224, width=224)

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
    dof_start = model.body_dofadr[body_id]

    # FIX: Lower linear velocity so it stays in frame, keep angular spin high for tumbling
    data.qvel[dof_start: dof_start + 3] = np.random.uniform(-0.5, 0.5, 3)
    data.qvel[dof_start + 3: dof_start + 6] = np.random.uniform(-12, 12, 3)

    mujoco.mj_forward(model, data)

    frames, labels, angular_velocities = [], [], []

    steady_wind = np.random.uniform(-0.8, 0.8, 3)

    for _ in range(120):  # Simulate 1.2 seconds
        # Apply the steady wind drift instead of chaotic jitter
        data.xfrc_applied[body_id][:3] = steady_wind

        mujoco.mj_step(model, data)
        renderer.update_scene(data, camera="main_cam")

        frames.append(renderer.render())
        u, v, z = project_3d_to_2d(model, data)
        labels.append([u, v, z])
        angular_velocities.append(np.linalg.norm(data.qvel[dof_start + 3: dof_start + 6]))

    # SLICE THE BEST 16-FRAME WINDOW
    best_start = 0
    max_energy = 0
    for i in range(len(frames) - 16):
        energy = sum(angular_velocities[i:i + 16])
        if energy > max_energy:
            max_energy = energy
            best_start = i

    best_frames = frames[best_start: best_start + 16]
    best_labels = labels[best_start: best_start + 16]

    # POST-PROCESSING (No more flickering!)
    # 1. Apply stable episode-level color shifts
    wb_frames = apply_episode_white_balance(best_frames)

    # 2. Apply mild per-frame sensor noise
    noisy_frames = [frame_noise(image=f)['image'] for f in wb_frames]

    final_video_tensor = np.array(noisy_frames, dtype=np.uint8)
    final_label_tensor = np.array(best_labels, dtype=np.float32)

    hdf5_file.create_dataset(f"video_{episode_idx}", data=final_video_tensor, compression="lzf")
    hdf5_file.create_dataset(f"label_{episode_idx}", data=final_label_tensor)

    print(f"Generated Episode {episode_idx} | Tumble Energy: {max_energy:.2f}")


if __name__ == "__main__":
    print("Initializing V2 Project STATERA Pipeline...")
    with h5py.File("statera_poc.hdf5", "w") as f:
        for i in range(5):
            generate_statera_episode(f, i)
    print("V2 Dataset Generation Complete.")