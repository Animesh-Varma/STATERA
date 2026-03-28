import mujoco
import numpy as np
import h5py
import albumentations as A
import random
from generator import generate_randomized_xml

frame_noise = A.Compose([
    A.MotionBlur(blur_limit=(7, 21), p=0.8),
    A.GaussNoise(std_range=(0.01, 0.04), p=0.7)
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
    com_world = data.xipos[body_id]

    cam_mat = data.cam_xmat[cam_id].reshape(3, 3)
    pt_cam = cam_mat.T @ (com_world - cam_pos)

    focal_length = 0.5 * resolution / np.tan(np.deg2rad(fovy) / 2)

    u = (pt_cam[0] / -pt_cam[2]) * focal_length + (resolution / 2)
    v = (resolution / 2) - (pt_cam[1] / -pt_cam[2]) * focal_length

    return float(u), float(v), float(-pt_cam[2])


def generate_statera_episode(dataset_file, episode_index):
    rand_mode = random.random()
    if rand_mode < 0.35:
        cam_mode = "STATIC"
    elif rand_mode < 0.675:
        cam_mode = "STABLE"
    else:
        cam_mode = "CHAOTIC"

    xml_string = generate_randomized_xml(cam_mode=cam_mode)
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

        if cam_mode == "STATIC":
            mid_z = data.xpos[body_id][2] * np.random.uniform(0.2, 0.8)
            lookat_offset = np.random.uniform(-0.6, 0.6, 3)
            data.mocap_pos[mocap_id] = np.array([0.0, 0.0, mid_z]) + lookat_offset
        else:
            # Aggressive offset for mobile tracking
            lookat_offset = np.random.uniform(-0.4, 0.4, 3)
            data.mocap_pos[mocap_id] = data.xpos[body_id].copy() + lookat_offset

        frames, labels, z_heights = [], [], []

        for step in range(3000):
            if step % 10 == 0:
                if cam_mode != "STATIC":
                    t = data.time
                    obj_pos = data.xpos[body_id]
                    cam_target = data.mocap_pos[mocap_id]

                    target_with_offset = obj_pos + lookat_offset

                    if cam_mode == "STABLE":
                        new_target = cam_target + 0.08 * (target_with_offset - cam_target)
                        jitter = np.zeros(3)
                    else:  # CHAOTIC
                        new_target = cam_target + 0.25 * (target_with_offset - cam_target)
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
            # Silent fail, will naturally retry
            return False, None, None

        # Randomize the impact frame placement within the 16-frame window (between index 3 and 12)
        # This inherently randomizes the start and end times of the video slice
        impact_offset = random.randint(3, 12)
        best_start = impact_frame - impact_offset

        # Make sure we actually have 16 valid frames after offsetting
        if best_start < 0 or best_start + 16 > len(frames):
            return False, None, None

        best_frames = frames[best_start: best_start + 16]
        best_labels = np.array(labels[best_start: best_start + 16], dtype=np.float32)

        uv_coords = best_labels[:, :2]
        z_depths = best_labels[:, 2:3]

        # --- NEW VALIDATION: CoM bounds and depth safety ---
        # If the CoM U/V coordinates leave the 384x384 image frame bounds, reject the episode.
        if np.any(uv_coords[:, 0] < 0) or np.any(uv_coords[:, 0] > 384):
            return False, None, None
        if np.any(uv_coords[:, 1] < 0) or np.any(uv_coords[:, 1] > 384):
            return False, None, None

        # Reject if the CoM gets dangerously close to the camera lens (closer than 25 cm).
        if np.any(z_depths < 0.25):
            return False, None, None

        wb_frames = apply_episode_white_balance(best_frames)
        noisy_frames = np.array([frame_noise(image=f)['image'] for f in wb_frames], dtype=np.uint8)

        noisy_frames_pytorch = np.transpose(noisy_frames, (0, 3, 1, 2))

        dataset_file["videos"][episode_index] = noisy_frames_pytorch
        dataset_file["uv_coords"][episode_index] = uv_coords
        dataset_file["z_depths"][episode_index] = z_depths

        print(f"Generated Ep {episode_index} | Cam: {cam_mode} | Impact Frame Index: {impact_offset}")
        return True, cam_mode, impact_offset

    finally:
        renderer.close()


if __name__ == "__main__":
    NUM_EPISODES = 5000
    print(f"Initializing STATERA Pipeline (PyTorch Structuring for {NUM_EPISODES} episodes)...")

    stats = {
        "attempts": 0,
        "successes": 0,
        "cam_modes": {"STATIC": 0, "STABLE": 0, "CHAOTIC": 0},
        "impact_frames": []
    }

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
            stats["attempts"] += 1
            success, mode, impact = generate_statera_episode(f, episodes_generated)
            if success:
                stats["successes"] += 1
                stats["cam_modes"][mode] += 1
                stats["impact_frames"].append(impact)
                episodes_generated += 1

        print("\nDataset Generation Complete. Compiling Statistical Validation Report...")

        # Pull the absolute ground truth straight from the finished HDF5 dataset
        all_uv = f["uv_coords"][:]
        all_z = f["z_depths"][:]

        # U (Horizontal), V (Vertical)
        mean_u, mean_v = np.mean(all_uv[:, :, 0]), np.mean(all_uv[:, :, 1])
        std_u, std_v = np.std(all_uv[:, :, 0]), np.std(all_uv[:, :, 1])
        min_u, max_u = np.min(all_uv[:, :, 0]), np.max(all_uv[:, :, 0])
        min_v, max_v = np.min(all_uv[:, :, 1]), np.max(all_uv[:, :, 1])

        mean_z = np.mean(all_z)
        min_z, max_z = np.min(all_z), np.max(all_z)

        avg_impact = np.mean(stats['impact_frames'])
        rejection_rate = 1.0 - (stats['successes'] / stats['attempts'])

        # Output the Report
        print("\n" + "=" * 60)
        print("STATERA DATASET GENERATION REPORT")
        print("=" * 60)
        print(f"Total Attempts   : {stats['attempts']}")
        print(f"Total Successes  : {stats['successes']} episodes")
        print(f"Rejection Rate   : {rejection_rate:.2%} (Out of bounds, no impact, etc.)")

        print("\n🎥 CAMERA MODE DISTRIBUTION:")
        for m, count in stats['cam_modes'].items():
            print(f"  - {m:<8}: {count} ({count / stats['successes']:.1%})")

        print(f"\n⚡ TEMPORAL VARIANCE:")
        print(f"  - Avg Impact Frame : {avg_impact:.2f} (Allowed Range: 3 - 12)")

        print("\n2D CENTER OF MASS SPREAD (Pixels):")
        print(f"  - U (Horizontal) : Mean = {mean_u:^6.1f} | Std = {std_u:^5.1f} | Range = [{min_u:.1f}, {max_u:.1f}]")
        print(f"  - V (Vertical)   : Mean = {mean_v:^6.1f} | Std = {std_v:^5.1f} | Range =[{min_v:.1f}, {max_v:.1f}]")

        print("\nZ-DEPTH SPREAD (Meters from Camera Lens):")
        print(f"  - Z-Depth        : Mean = {mean_z:^6.2f} | Range =[{min_z:.2f}, {max_z:.2f}]")
        print("=" * 60 + "\n")