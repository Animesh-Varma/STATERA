import mujoco
import numpy as np
import h5py
import albumentations as A
import random
from generator import generate_randomized_xml

# Domain Randomization pipeline
frame_noise = A.Compose([
    A.MotionBlur(blur_limit=(3, 7), p=0.5),
    A.GaussNoise(std_range=(0.01, 0.04), p=0.8)  # This is correct for Albumentations v2.0+
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


def project_3d_to_2d(model, data, camera_name="main_cam", resolution=384):
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    cam_pos = data.cam_xpos[cam_id]
    fovy = model.cam_fovy[cam_id]

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
    com_world = data.xipos[body_id] # data.xipos is the true Cartesian CoM

    # Get the true Rotation Matrix from Camera to World (3x3)
    # MuJoCo updates this natively during mj_step/mj_forward, even for targetbody
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3)

    # Transform Center of Mass to Camera space (cam_mat.T is World-to-Camera)
    pt_cam = cam_mat.T @ (com_world - cam_pos)

    focal_length = 0.5 * resolution / np.tan(np.deg2rad(fovy) / 2)

    # MuJoCo cameras look down the -Z axis. X is right, Y is up.
    # U goes right, V goes down (Image coordinates)
    u = (pt_cam[0] / -pt_cam[2]) * focal_length + (resolution / 2)
    v = (resolution / 2) - (pt_cam[1] / -pt_cam[2]) * focal_length

    # Returns U, V, and the Positive Depth from the camera lens
    return float(u), float(v), float(-pt_cam[2])


def generate_statera_episode(dataset_file, episode_index):
    is_stable = random.random() < 0.5

    xml_string = generate_randomized_xml(is_stable=is_stable)
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=384, width=384)

    try:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
        tracker_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "camera_tracker")
        mocap_id = model.body_mocapid[tracker_id]
        dof_start = model.body_dofadr[body_id]

        data.qvel[dof_start: dof_start + 3] = np.random.uniform(-1.0, 1.0, 3)
        data.qvel[dof_start + 3: dof_start + 6] = np.random.uniform(-10.0, 10.0, 3)

        mujoco.mj_forward(model, data)
        data.mocap_pos[mocap_id] = data.xpos[body_id].copy()

        frames, labels, z_heights = [], [], []

        for step in range(3000):
            if step % 10 == 0:
                t = data.time
                obj_pos = data.xpos[body_id]
                cam_target = data.mocap_pos[mocap_id]

                if is_stable:
                    new_target = cam_target + 0.08 * (obj_pos - cam_target)
                    jitter = np.zeros(3)
                else:
                    new_target = cam_target + 0.25 * (obj_pos - cam_target)
                    jitter = np.array([
                        0.010 * np.sin(10 * t) + 0.005 * np.cos(23 * t),
                        0.015 * np.cos(12 * t) + 0.005 * np.sin(27 * t),
                        0.010 * np.sin(15 * t)
                    ])

                data.mocap_pos[mocap_id] = new_target + jitter

            mujoco.mj_step(model, data)

            if step % 42 == 0:
                renderer.update_scene(data, camera="main_cam")
                frames.append(renderer.render().copy())

                u, v, z = project_3d_to_2d(model, data)
                labels.append([u, v, z])
                z_heights.append(data.xipos[body_id][2])

        impact_detected = False
        impact_frame = 0
        is_falling = False

        for i in range(1, len(z_heights)):
            dz = z_heights[i] - z_heights[i - 1]
            if dz < -0.015:
                is_falling = True
            elif is_falling and dz > -0.002:
                impact_frame = i
                impact_detected = True
                break

        if not impact_detected:
            print(f"Ep {episode_index} FAILED: No valid impact. Retrying...")
            return False  # Return False to trigger a retry loop

        best_start = max(0, min(impact_frame - 4, len(frames) - 16))

        best_frames = frames[best_start: best_start + 16]
        best_labels = np.array(labels[best_start: best_start + 16], dtype=np.float32)

        assert len(best_frames) == 16, f"Data size mismatch. Expected 16 frames, got {len(best_frames)}."

        wb_frames = apply_episode_white_balance(best_frames)
        noisy_frames = np.array([frame_noise(image=f)['image'] for f in wb_frames], dtype=np.uint8)

        # Convert shape from (16, 384, 384, 3) to PyTorch layout (16, 3, 384, 384)
        noisy_frames_pytorch = np.transpose(noisy_frames, (0, 3, 1, 2))

        # Slice the labels into the exact structures requested
        uv_coords = best_labels[:, :2]  # Shape: (16, 2)
        z_depths = best_labels[:, 2:3]  # Shape: (16, 1)

        # Insert directly into the pre-allocated HDF5 monolithic arrays
        dataset_file["videos"][episode_index] = noisy_frames_pytorch
        dataset_file["uv_coords"][episode_index] = uv_coords
        dataset_file["z_depths"][episode_index] = z_depths

        cam_mode = "STABLE" if is_stable else "CHAOTIC"
        print(
            f"Generated Ep {episode_index} | Cam: {cam_mode} | Impact Frame: {impact_frame} | Captured: {best_start} to {best_start + 16}")
        return True

    finally:
        renderer.close()


if __name__ == "__main__":
    NUM_EPISODES = 2000  # Audit target. Full PoC target: 2000 episodes (Train: 1600, Val: 400)
    print(f"Initializing STATERA Pipeline (PyTorch Structuring for {NUM_EPISODES} episodes)...")

    with h5py.File("statera_poc.hdf5", "w") as f:
        f.create_dataset("videos",
                         shape=(NUM_EPISODES, 16, 3, 384, 384),
                         dtype=np.uint8,
                         compression="lzf",
                         chunks=(1, 16, 3, 384, 384))

        f.create_dataset("uv_coords",
                         shape=(NUM_EPISODES, 16, 2),
                         dtype=np.float32,
                         chunks=(1, 16, 2))

        f.create_dataset("z_depths",
                         shape=(NUM_EPISODES, 16, 1),
                         dtype=np.float32,
                         chunks=(1, 16, 1))


        episodes_generated = 0
        while episodes_generated < NUM_EPISODES:
            success = generate_statera_episode(f, episodes_generated)
            if success is not False:
                episodes_generated += 1

    print("Dataset Generation Complete.")