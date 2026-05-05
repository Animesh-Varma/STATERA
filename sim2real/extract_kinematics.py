import cv2
import numpy as np
import json
import os
import glob
import math
import h5py
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R

def get_var(key, prompt_msg, cast_type, default):
    config_file = 'variables.txt'
    config = {}
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
        except: pass

    if key not in config:
        val = input(f"{prompt_msg} (Default: {default}) -> ").strip()
        if not val:
            config[key] = default
        else:
            try:
                if cast_type == list:
                    config[key] =[float(x.strip()) for x in val.split(',')]
                else:
                    config[key] = cast_type(val)
            except Exception:
                print(f"[!] Invalid input. Using default: {default}")
                config[key] = default

        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)

    return config[key]


def apply_savgol_kinematics(rvecs, tvecs, window_length=9, polyorder=3):
    """
    Applies independent linear smoothing to Translations, and
    mathematically safe Quaternion smoothing to Rotations.
    """
    # 1. Smooth Translations linearly
    smoothed_tvecs = np.zeros_like(tvecs)
    for j in range(3):
        smoothed_tvecs[:, j] = savgol_filter(tvecs[:, j], window_length, polyorder)

    # 2. Smooth Rotations using Quaternions
    rotations = R.from_rotvec(rvecs)
    quats = rotations.as_quat()  # Returns[x, y, z, w]

    # Enforce Quaternion continuity (prevent hemisphere flipping step-functions)
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[i-1]) < 0:
            quats[i] = -quats[i]

    # Smooth the continuous quaternion components
    smoothed_quats = np.zeros_like(quats)
    for j in range(4):
        smoothed_quats[:, j] = savgol_filter(quats[:, j], window_length, polyorder)

    # Re-normalize quaternions to restore valid 3D rotations!
    norms = np.linalg.norm(smoothed_quats, axis=1, keepdims=True)
    smoothed_quats /= norms

    # Convert back to OpenCV Rodrigues vectors
    smoothed_rvecs = R.from_quat(smoothed_quats).as_rotvec()

    return smoothed_rvecs, smoothed_tvecs


def clamp_com_to_polygon(pt, polygon):
    pt_tuple = (float(pt[0]), float(pt[1]))
    if cv2.pointPolygonTest(polygon, pt_tuple, False) >= 0:
        return pt_tuple

    closest_pt = None
    min_dist = float('inf')
    n = len(polygon)

    pt_arr = np.array(pt_tuple)
    for i in range(n):
        a = polygon[i][0]
        b = polygon[(i+1)%n][0]

        ab = b - a
        ap = pt_arr - a
        dot_ab = np.dot(ab, ab)
        t = 0.0 if dot_ab == 0 else np.dot(ap, ab) / dot_ab
        t = max(0.0, min(1.0, t))
        cp = a + t * ab

        dist = np.linalg.norm(pt_arr - cp)
        if dist < min_dist:
            min_dist = dist
            closest_pt = cp

    return closest_pt


def process_directory():
    print("="*60)
    print(" STATERA: Kinematics Batch Extraction Setup (PHYSICS PATCHED)")
    print("="*60)

    marker_size_cm = get_var('marker_size_cm', 'Target black marker size in cm', float, 10.0)
    box_size_cm = get_var('box_size_cm', 'Box dimensions X, Y, Z in cm', list,[15.0, 15.0, 15.0])
    mujoco_com_offset_cm = get_var('com_offset_cm', 'MuJoCo Hidden CoM offset in cm', list,[4.0, -3.0, 5.0])

    max_frames = get_var('max_frames', 'Number of strictly disjoint test frames per sequence', int, 16)
    stride = get_var('stride', 'Sliding Window Stride (Recommend 16 for testing)', int, 16)
    buffer_pad = get_var('buffer_pad', 'Padding frames on each side for Sav-Gol (Recommend 4)', int, 4)

    total_extract_len = max_frames + (buffer_pad * 2)

    input_dir = get_var('input_dir', 'Path to input video directory', str, 'data')
    output_dir = get_var('output_dir', 'Path to output HDF5 directory', str, 'output')

    os.makedirs(output_dir, exist_ok=True)
    video_files = glob.glob(os.path.join(input_dir, '*.mp4'))

    if not video_files:
        print(f"[!] No .mp4 files found in '{input_dir}'. Exiting.")
        return

    mj_x, mj_y, mj_z =[v / 100.0 for v in mujoco_com_offset_cm]
    cv_x, cv_y, cv_z = -mj_x, -mj_z, mj_y

    opencv_com_offset_m =[cv_x, cv_y, cv_z]
    true_com_mag_m = math.sqrt(cv_x**2 + cv_y**2 + cv_z**2)

    bx, by, bz =[v / 100.0 for v in box_size_cm]
    hx, hy, hz = bx / 2, by / 2, bz / 2
    hm = (marker_size_cm / 100.0) / 2

    DYNAMIC_MARKERS = {
        0: [[-hm, -hm,  hz],[ hm, -hm,  hz], [ hm,  hm,  hz],[-hm,  hm,  hz]],
        1: [[ hm, -hm, -hz],[-hm, -hm, -hz],[-hm,  hm, -hz], [ hm,  hm, -hz]],
        2: [[-hx, -hm, -hm], [-hx, -hm,  hm],[-hx,  hm,  hm],[-hx,  hm, -hm]],
        3: [[ hx, -hm,  hm],[ hx, -hm, -hm], [ hx,  hm, -hm],[ hx,  hm,  hm]],
        4: [[-hm, -hy, -hm],[ hm, -hy, -hm],[ hm, -hy,  hm],[-hm, -hy,  hm]],
        5: [[-hm,  hy,  hm], [ hm,  hy,  hm],[ hm,  hy, -hm], [-hm,  hy, -hm]]
    }

    print("\n[INFO] Loading Intrinsic Parameters...")
    try:
        mtx = np.load('camera_matrix.npy')
        dist = np.load('dist_coeffs.npy')
    except FileNotFoundError:
        print("[!] camera_matrix.npy or dist_coeffs.npy missing. Exiting.")
        return

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    box_3d = np.array([[-hx, -hy, -hz],[hx, -hy, -hz], [hx, hy, -hz],[-hx, hy, -hz],
        [-hx, -hy,  hz],[hx, -hy,  hz],[hx, hy,  hz], [-hx, hy,  hz]
    ], dtype=np.float32)

    com_3d = np.array([opencv_com_offset_m], dtype=np.float32)

    for v_idx, video_path in enumerate(video_files):
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"\n[->] Scrubbing Video {v_idx + 1}/{len(video_files)}: {base_name}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): continue

        out_h5_path = os.path.join(output_dir, f"{base_name}_statera.h5")
        h5_f = h5py.File(out_h5_path, 'w')

        h5_f.attrs["source_video"] = base_name
        h5_f.attrs["box_size_m"] =[bx, by, bz]
        h5_f.attrs["hidden_com_m"] = opencv_com_offset_m
        h5_f.attrs["com_magnitude_m"] = true_com_mag_m
        h5_f.attrs["resolution"] =[384, 384]

        raw_buffer =[]
        original_frame_count = 0
        seq_idx = 0
        frame_h, frame_w = 0, 0

        while True:
            ret, frame = cap.read()
            if not ret: break

            if frame_h == 0:
                frame_h, frame_w = frame.shape[:2]

            corners, ids, _ = detector.detectMarkers(frame)
            detection_success = False

            if ids is not None and len(ids) > 0:
                obj_points, img_points = [],[]
                for i, marker_id in enumerate(ids):
                    mid = int(marker_id[0])
                    if mid in DYNAMIC_MARKERS:
                        for j in range(4):
                            obj_points.append(DYNAMIC_MARKERS[mid][j])
                            img_points.append(corners[i][0][j])

                if len(img_points) >= 4:
                    raw_buffer.append({
                        "frame": frame.copy(),
                        "obj_pts": np.array(obj_points, dtype=np.float32),
                        "img_pts": np.array(img_points, dtype=np.float32),
                        "original_idx": original_frame_count
                    })
                    detection_success = True

            if not detection_success:
                if len(raw_buffer) > 0:
                    print(f"   [!] Lost tracking at frame {original_frame_count}. Resetting streak...")
                raw_buffer =[]

            if len(raw_buffer) == total_extract_len:
                anchor_idx = 0
                max_pts = 0
                for idx, item in enumerate(raw_buffer):
                    if len(item['obj_pts']) > max_pts:
                        max_pts = len(item['obj_pts'])
                        anchor_idx = idx

                poses = [None] * total_extract_len

                # Solve Anchor
                _, r_anch, t_anch = cv2.solvePnP(raw_buffer[anchor_idx]['obj_pts'], raw_buffer[anchor_idx]['img_pts'], mtx, dist, flags=cv2.SOLVEPNP_SQPNP)
                poses[anchor_idx] = (r_anch, t_anch)

                for i in range(anchor_idx + 1, total_extract_len):
                    guess_r, guess_t = poses[i-1][0].copy(), poses[i-1][1].copy()
                    _, r, t = cv2.solvePnP(raw_buffer[i]['obj_pts'], raw_buffer[i]['img_pts'], mtx, dist,
                                           rvec=guess_r, tvec=guess_t, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
                    poses[i] = (r, t)

                # Track Backwards
                for i in range(anchor_idx - 1, -1, -1):
                    guess_r, guess_t = poses[i+1][0].copy(), poses[i+1][1].copy()
                    _, r, t = cv2.solvePnP(raw_buffer[i]['obj_pts'], raw_buffer[i]['img_pts'], mtx, dist,
                                           rvec=guess_r, tvec=guess_t, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
                    poses[i] = (r, t)

                rvecs = np.array([p[0].flatten() for p in poses])
                tvecs = np.array([p[1].flatten() for p in poses])

                # Proper Quaternion Math Smoothing applied here
                rvecs, tvecs = apply_savgol_kinematics(rvecs, tvecs, window_length=9, polyorder=3)

                test_rvecs = rvecs[buffer_pad : buffer_pad + max_frames]
                test_tvecs = tvecs[buffer_pad : buffer_pad + max_frames]
                test_raw_buffer = raw_buffer[buffer_pad : buffer_pad + max_frames]

                temp_buffer =[]
                all_xs, all_ys = [],[]

                for i in range(max_frames):
                    rvec = test_rvecs[i].reshape(3,1)
                    tvec = test_tvecs[i].reshape(3,1)

                    box_2d, _ = cv2.projectPoints(box_3d, rvec, tvec, mtx, dist)
                    com_2d, _ = cv2.projectPoints(com_3d, rvec, tvec, mtx, dist)

                    all_xs.extend(box_2d[:, 0, 0])
                    all_ys.extend(box_2d[:, 0, 1])

                    hull = cv2.convexHull(box_2d)
                    raw_com = com_2d[0][0]
                    clamped_com_u, clamped_com_v = clamp_com_to_polygon(raw_com, hull)

                    temp_buffer.append({
                        "clean_frame": test_raw_buffer[i]["frame"],
                        "com_u": clamped_com_u,
                        "com_v": clamped_com_v,
                        "rvec": rvec.flatten(),
                        "tvec": tvec.flatten(),
                        "original_idx": test_raw_buffer[i]["original_idx"]
                    })

                cx, cy = (int(np.min(all_xs)) + int(np.max(all_xs))) // 2, (int(np.min(all_ys)) + int(np.max(all_ys))) // 2
                target_size = min(frame_w, frame_h)
                half_size = target_size // 2
                x1, y1 = cx - half_size, cy - half_size

                if x1 < 0: x1 = 0
                elif x1 + target_size > frame_w: x1 = frame_w - target_size

                if y1 < 0: y1 = 0
                elif y1 + target_size > frame_h: y1 = frame_h - target_size

                x2, y2 = x1 + target_size, y1 + target_size
                scale = 384.0 / target_size

                seq_frames, seq_u, seq_v, seq_z, seq_rvec, seq_tvec, seq_orig = [], [], [],[], [], [],[]

                for i, item in enumerate(temp_buffer):
                    crop_window = item["clean_frame"][y1:y2, x1:x2]
                    final_frame = cv2.resize(crop_window, (384, 384), interpolation=cv2.INTER_AREA)

                    final_com_u = (item["com_u"] - x1) * scale
                    final_com_v = (item["com_v"] - y1) * scale

                    seq_frames.append(final_frame)
                    seq_u.append(final_com_u)
                    seq_v.append(final_com_v)
                    seq_z.append(item["tvec"][2])
                    seq_rvec.append(item["rvec"])
                    seq_tvec.append(item["tvec"])
                    seq_orig.append(item["original_idx"])

                grp = h5_f.create_group(f"seq_{seq_idx:04d}")
                grp.create_dataset("frames", data=np.array(seq_frames, dtype=np.uint8), compression="lzf")
                grp.create_dataset("com_u", data=np.array(seq_u, dtype=np.float32))
                grp.create_dataset("com_v", data=np.array(seq_v, dtype=np.float32))
                grp.create_dataset("z_depth_meters", data=np.array(seq_z, dtype=np.float32))
                grp.create_dataset("rvecs", data=np.array(seq_rvec, dtype=np.float32))
                grp.create_dataset("tvecs", data=np.array(seq_tvec, dtype=np.float32))
                grp.create_dataset("original_video_frame", data=np.array(seq_orig, dtype=np.int32))

                print(f"   [OK] Evaluated purely disjoint test set {seq_idx:04d}")

                raw_buffer = raw_buffer[stride:]
                seq_idx += 1

            original_frame_count += 1

        cap.release()
        h5_f.close()
        print(f"   [SUCCESS] Saved {seq_idx} pure test sequences to database: {out_h5_path}")

    print("\n[INFO] Batch processing complete!")

if __name__ == "__main__":
    process_directory()