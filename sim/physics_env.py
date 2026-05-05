import mujoco
import numpy as np
import h5py
import albumentations as A
import random
import wandb
from generator import generate_randomized_xml

# Simulating hardware sensor artifacts to condition the feature extractor for downstream SQPnP execution
frame_noise = A.Compose([
    A.MotionBlur(blur_limit=(7, 21), p=0.8),
    A.GaussNoise(std_range=(0.01, 0.04), p=0.7)
])


def apply_episode_white_balance(frames):
    """
    Simulates global lighting fluctuations across the image sequence.
    This acts as dynamic exposure data augmentation, fortifying the 1D temporal convolution
    against temporally inconsistent luminosity inputs.
    """
    contrast = np.random.uniform(0.8, 1.2)
    brightness = np.random.uniform(-30, 30)
    r_shift, g_shift, b_shift = np.random.uniform(-20, 20, 3)

    processed =[]
    for f in frames:
        f = f.astype(np.float32) * contrast + brightness
        f[:, :, 0] += r_shift
        f[:, :, 1] += g_shift
        f[:, :, 2] += b_shift
        processed.append(np.clip(f, 0, 255).astype(np.uint8))
    return processed


def project_point_to_2d(model, data, point_3d, camera_name="main_cam", resolution=640):
    """
    Projects 3D world coordinates to the 2D image plane using a strict pinhole camera model.
    These absolute projections serve as the mathematical ground truth to evaluate the accuracy
    of our SQPnP solver when reconstructing the 3D Center of Mass from 2D visual predictions.
    """
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    cam_pos = data.cam_xpos[cam_id]
    fovy = model.cam_fovy[cam_id]

    cam_mat = data.cam_xmat[cam_id].reshape(3, 3)
    pt_cam = cam_mat.T @ (point_3d - cam_pos)

    focal_length = 0.5 * resolution / np.tan(np.deg2rad(fovy) / 2)

    u = (pt_cam[0] / -pt_cam[2]) * focal_length + (resolution / 2)
    v = (resolution / 2) - (pt_cam[1] / -pt_cam[2]) * focal_length

    return float(u), float(v), float(-pt_cam[2])


def get_projected_bbox_area(model, data, camera_name="main_cam", resolution=640):
    """
    Computes screen coverage to autonomously reject degenerate trajectory configurations
    (e.g., lens clipping) that would otherwise poison the SQPnP keypoint distribution.
    """
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "shell")
    geom_type = model.geom_type[geom_id]
    size = model.geom_size[geom_id]

    if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        sx, sy, sz = size[0], size[0], size[1]
    else:
        sx, sy, sz = size[0], size[1], size[2]
        if sx == 0 and sy == 0 and sz == 0:
            # Fallback AABB for custom meshes
            sx = sy = sz = model.geom_rbound[geom_id]

    corners_local = np.array([
        [ sx,  sy,  sz], [ sx,  sy, -sz],[ sx, -sy,  sz], [ sx, -sy, -sz],[-sx,  sy,  sz], [-sx,  sy, -sz], [-sx, -sy,  sz],[-sx, -sy, -sz]
    ])

    geom_pos = data.geom_xpos[geom_id]
    geom_mat = data.geom_xmat[geom_id].reshape(3, 3)
    corners_world = geom_pos + corners_local @ geom_mat.T

    us, vs = [],[]
    for pt in corners_world:
        u, v, z = project_point_to_2d(model, data, pt, camera_name=camera_name, resolution=resolution)
        if z <= 0.15:
            return float('inf')
        us.append(u)
        vs.append(v)

    area = (max(us) - min(us)) * (max(vs) - min(vs))
    return area


def generate_statera_episode(dataset_file, episode_index):
    """
    Orchestrates the generation of an individual training episode.
    It simulates physics, captures frames, isolates the critical impact event,
    and slices the exact T=16 sequence length required as input for our
    1D temporal convolution module.
    """
    rand_mode = random.random()
    if rand_mode < 0.35:
        cam_mode = "STATIC"
    elif rand_mode < 0.675:
        cam_mode = "STABLE"
    else:
        cam_mode = "CHAOTIC"

    xml_string, com_mag = generate_randomized_xml(cam_mode=cam_mode)
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)

    renderer = mujoco.Renderer(model, height=640, width=640)

    try:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
        tracker_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "camera_tracker")
        mocap_id = model.body_mocapid[tracker_id]
        dof_start = model.body_dofadr[body_id]

        data.qvel[dof_start: dof_start + 3] = np.random.uniform(-1.0, 1.0, 3)
        data.qvel[dof_start + 3: dof_start + 6] = np.random.uniform(-25.0, 25.0, 3)

        mujoco.mj_forward(model, data)

        if cam_mode == "STATIC":
            mid_z = data.xpos[body_id][2] * np.random.uniform(0.2, 0.8)
            lookat_offset = np.random.uniform(-0.6, 0.6, 3)
            data.mocap_pos[mocap_id] = np.array([0.0, 0.0, mid_z]) + lookat_offset
        else:
            lookat_offset = np.random.uniform(-0.4, 0.4, 3)
            data.mocap_pos[mocap_id] = data.xpos[body_id].copy() + lookat_offset

        frames, labels, geom_centers, z_heights = [], [], [],[]

        for step in range(3000):
            # Controls temporal tracking updates. Subsampling ensures non-trivial inter-frame dynamics
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
                        # Applies multi-frequency chaotic jitter to benchmark the 1D temporal
                        # convolution's resilience to high-frequency domain perturbations.
                        new_target = cam_target + 0.25 * (target_with_offset - cam_target)
                        jitter = np.array([
                            0.010 * np.sin(10 * t) + 0.005 * np.cos(23 * t),
                            0.015 * np.cos(12 * t) + 0.005 * np.sin(27 * t),
                            0.010 * np.sin(15 * t)
                        ])
                    data.mocap_pos[mocap_id] = new_target + jitter

            mujoco.mj_step(model, data)

            if data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0 or np.any(np.isnan(data.qvel)):
                return False, None, None, None, None

            # Frame extraction frequency mapping to the required temporal convolution sampling rate
            if step % 42 == 0:
                # Dynamic rejection if the object exceeds 75% of a 384x384 box (110,592 pixels)
                bbox_area = get_projected_bbox_area(model, data, resolution=640)
                if bbox_area > 110592:
                    return False, None, None, None, None

                renderer.update_scene(data, camera="main_cam")
                frames.append(renderer.render().copy())

                com_world = data.xipos[body_id]
                u_com, v_com, z_com = project_point_to_2d(model, data, com_world, resolution=640)
                labels.append([u_com, v_com, z_com])

                geom_world = data.xpos[body_id]
                u_geom, v_geom, z_geom = project_point_to_2d(model, data, geom_world, resolution=640)
                geom_centers.append([u_geom, v_geom, z_geom])

                z_heights.append(com_world[2])

        impact_detected = False
        impact_frame = 0
        is_falling = False

        # Analyzes vertical velocity deltas to autonomously tag the primary impact frame
        for i in range(1, len(z_heights)):
            dz = z_heights[i] - z_heights[i - 1]
            if dz < -0.015:
                is_falling = True
            elif is_falling and dz > -0.002:
                impact_frame = i
                impact_detected = True
                break

        if not impact_detected:
            return False, None, None, None, None

        impact_offset = random.randint(3, 12)
        best_start = impact_frame - impact_offset

        # Strict enforcement of the T=16 contiguous sequence length required by the architecture
        if best_start < 0 or best_start + 16 > len(frames):
            return False, None, None, None, None

        best_frames = frames[best_start: best_start + 16]
        best_labels = np.array(labels[best_start: best_start + 16], dtype=np.float32)
        best_geoms = np.array(geom_centers[best_start: best_start + 16], dtype=np.float32)

        uv_coords = best_labels[:, :2]
        z_depths = best_labels[:, 2:3]
        box_center_uv = best_geoms[:, :2]

        min_u, max_u = np.min(uv_coords[:, 0]), np.max(uv_coords[:, 0])
        min_v, max_v = np.min(uv_coords[:, 1]), np.max(uv_coords[:, 1])

        if (max_u - min_u) > 360 or (max_v - min_v) > 360:
            return False, None, None, None, None

        if random.random() < 0.5:
            crop_u = int((min_u + max_u) / 2.0 - 192)
            crop_v = int((min_v + max_v) / 2.0 - 192)
        else:
            crop_u = (640 - 384) // 2
            crop_v = (640 - 384) // 2

        crop_u = max(0, min(crop_u, 640 - 384))
        crop_v = max(0, min(crop_v, 640 - 384))

        cropped_frames = [f[crop_v:crop_v + 384, crop_u:crop_u + 384, :] for f in best_frames]

        uv_coords[:, 0] -= crop_u
        uv_coords[:, 1] -= crop_v

        box_center_uv[:, 0] -= crop_u
        box_center_uv[:, 1] -= crop_v

        if np.any(uv_coords[:, 0] < 0) or np.any(uv_coords[:, 0] > 384):
            return False, None, None, None, None
        if np.any(uv_coords[:, 1] < 0) or np.any(uv_coords[:, 1] > 384):
            return False, None, None, None, None
        if np.any(z_depths < 0.25):
            return False, None, None, None, None

        wb_frames = apply_episode_white_balance(cropped_frames)
        noisy_frames = np.array([frame_noise(image=f)['image'] for f in wb_frames], dtype=np.uint8)

        noisy_frames_pytorch = np.transpose(noisy_frames, (0, 3, 1, 2))

        dataset_file["videos"][episode_index] = noisy_frames_pytorch
        dataset_file["uv_coords"][episode_index] = uv_coords
        dataset_file["z_depths"][episode_index] = z_depths
        dataset_file["com_magnitudes"][episode_index] = com_mag
        dataset_file["box_center_uv"][episode_index] = box_center_uv

        print(f"Generated Ep {episode_index} | Cam: {cam_mode} | Impact Frame Index: {impact_offset}")
        return True, cam_mode, impact_offset, com_mag, noisy_frames_pytorch

    finally:
        renderer.close()


if __name__ == "__main__":
    NUM_EPISODES = 1000
    print(f"Initializing STATERA Pipeline (PyTorch Structuring for {NUM_EPISODES} episodes)...")

    wandb.init(project="STATERA-DataGen", job_type="generation")

    stats = {
        "attempts": 0,
        "successes": 0,
        "cam_modes": {"STATIC": 0, "STABLE": 0, "CHAOTIC": 0},
        "impact_frames":[]
    }

    # HDF5 compilation specifically structured for highly optimized sequential data loading
    # during the 1D Temporal Convolution Network's PyTorch dataloader phase.
    with h5py.File("1K-ablation.hdf5", "w") as f:
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

        f.create_dataset("com_magnitudes",
                         shape=(NUM_EPISODES, 1),
                         dtype=np.float32,
                         chunks=(1, 1))

        f.create_dataset("box_center_uv",
                         shape=(NUM_EPISODES, 16, 2),
                         dtype=np.float32,
                         chunks=(1, 16, 2))

        episodes_generated = 0
        video_buffer = []
        com_mag_list =[]

        while episodes_generated < NUM_EPISODES:
            stats["attempts"] += 1
            success, mode, impact, com_mag, frames_pt = generate_statera_episode(f, episodes_generated)
            if success:
                stats["successes"] += 1
                stats["cam_modes"][mode] += 1
                stats["impact_frames"].append(impact)
                com_mag_list.append(float(com_mag))

                if episodes_generated % 1000 < 5:
                    video_buffer.append(frames_pt)

                episodes_generated += 1

                if episodes_generated % 1000 == 0:
                    wandb.log({
                        "video_samples":[wandb.Video(v, fps=10, format="mp4") for v in video_buffer],
                        "step": episodes_generated
                    })
                    video_buffer =[]

        wandb.log({"com_magnitude_distribution": wandb.Histogram(com_mag_list)})

        print("\nDataset Generation Complete. Compiling Statistical Validation Report...")

        all_uv = f["uv_coords"][:]
        all_z = f["z_depths"][:]

        mean_u, mean_v = np.mean(all_uv[:, :, 0]), np.mean(all_uv[:, :, 1])
        std_u, std_v = np.std(all_uv[:, :, 0]), np.std(all_uv[:, :, 1])
        min_u, max_u = np.min(all_uv[:, :, 0]), np.max(all_uv[:, :, 0])
        min_v, max_v = np.min(all_uv[:, :, 1]), np.max(all_uv[:, :, 1])

        mean_z = np.mean(all_z)
        min_z, max_z = np.min(all_z), np.max(all_z)

        avg_impact = np.mean(stats['impact_frames'])
        rejection_rate = 1.0 - (stats['successes'] / stats['attempts'])

        print("\n" + "=" * 60)
        print("STATERA DATASET GENERATION REPORT")
        print("=" * 60)
        print(f"Total Attempts   : {stats['attempts']}")
        print(f"Total Successes  : {stats['successes']} episodes")
        print(f"Rejection Rate   : {rejection_rate:.2%} (Screen coverage >75%, OOB, NaN catches)")

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

    wandb.finish()