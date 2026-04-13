"""
3_extract_kinematics_batch.py
Sim-to-Real Dataset Generator.
*REFINED*:
- Replaces sliding-window smoothing with Global Newtonian Polynomial Regression.
- Mathematically forces a perfect physical curve (Degree-2 Parabola) over the 16 frames.
- Completely eradicates "first-frame hooks" and all sub-pixel high-frequency jitter.
"""

import cv2
import numpy as np
import json
import os
import glob
import math

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


def apply_newtonian_polyfit(rvecs, tvecs, degree=2):
    """
    Absolute Kinematic Smoothing.
    Over a 16-frame sequence, gravity/inertia dictate that 3D translation and rotation
    MUST follow a polynomial curve. We fit a global curve through the noisy points.
    """
    frames = np.arange(len(rvecs))
    smoothed_rvecs = np.zeros_like(rvecs)
    smoothed_tvecs = np.zeros_like(tvecs)

    for j in range(3):
        # Fit polynomial for Translation (Inertia/Gravity Parabola)
        t_coeffs = np.polyfit(frames, tvecs[:, j], degree)
        smoothed_tvecs[:, j] = np.polyval(t_coeffs, frames)

        # Fit polynomial for Rotation (Angular Velocity Arc)
        r_coeffs = np.polyfit(frames, rvecs[:, j], degree)
        smoothed_rvecs[:, j] = np.polyval(r_coeffs, frames)

    return smoothed_rvecs, smoothed_tvecs


def process_directory():
    print("="*60)
    print(" STATERA: Kinematics Batch Extraction Setup")
    print("="*60)

    marker_size_cm = get_var('marker_size_cm', 'Target black marker size in cm', float, 10.0)
    box_size_cm = get_var('box_size_cm', 'Box dimensions X, Y, Z in cm', list,[15.0, 15.0, 15.0])
    mujoco_com_offset_cm = get_var('com_offset_cm', 'MuJoCo Hidden CoM offset (X=Right, Y=Fwd, Z=Up) in cm', list,[4.0, -3.0, 5.0])
    max_frames = get_var('max_frames', 'Number of continuous frames to output per video', int, 16)

    input_dir = get_var('input_dir', 'Path to input video directory', str, 'data')
    output_dir = get_var('output_dir', 'Path to output directory', str, 'output')

    os.makedirs(output_dir, exist_ok=True)
    video_files = glob.glob(os.path.join(input_dir, '*.mp4'))

    if not video_files:
        print(f"[!] No .mp4 files found in '{input_dir}'. Exiting.")
        return

    mj_x, mj_y, mj_z =[v / 100.0 for v in mujoco_com_offset_cm]
    cv_x = -mj_x
    cv_y = -mj_z
    cv_z = mj_y

    opencv_com_offset_m =[cv_x, cv_y, cv_z]
    true_com_mag_m = math.sqrt(cv_x**2 + cv_y**2 + cv_z**2)

    bx, by, bz =[v / 100.0 for v in box_size_cm]
    hx, hy, hz = bx / 2, by / 2, bz / 2
    hm = (marker_size_cm / 100.0) / 2

    DYNAMIC_MARKERS = {
        0: [[-hm, -hm,  hz],[ hm, -hm,  hz],[ hm,  hm,  hz], [-hm,  hm,  hz]],
        1: [[ hm, -hm, -hz], [-hm, -hm, -hz],[-hm,  hm, -hz],[ hm,  hm, -hz]],
        2: [[-hx, -hm, -hm],[-hx, -hm,  hm], [-hx,  hm,  hm],[-hx,  hm, -hm]],
        3: [[ hx, -hm,  hm],[ hx, -hm, -hm],[ hx,  hm, -hm], [ hx,  hm,  hm]],
        4: [[-hm, -hy, -hm], [ hm, -hy, -hm],[ hm, -hy,  hm], [-hm, -hy,  hm]],
        5: [[-hm,  hy,  hm],[ hm,  hy,  hm], [ hm,  hy, -hm],[-hm,  hy, -hm]]
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

    box_3d = np.array([
        [-hx, -hy, -hz],[hx, -hy, -hz],[hx, hy, -hz], [-hx, hy, -hz],[-hx, -hy,  hz],[hx, -hy,  hz], [hx, hy,  hz],[-hx, hy,  hz]
    ], dtype=np.float32)

    com_3d = np.array([opencv_com_offset_m], dtype=np.float32)

    for v_idx, video_path in enumerate(video_files):
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"\n[->] Scrubbing Video {v_idx + 1}/{len(video_files)}: {base_name}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): continue

        out_video_path = os.path.join(output_dir, f"{base_name}_statera.mp4")
        out_json_path = os.path.join(output_dir, f"{base_name}_benchmark.json")

        statera_data = {
            "metadata": {
                "source_video": base_name,
                "box_size_m": [bx, by, bz],
                "hidden_com_m": opencv_com_offset_m,
                "com_magnitude_m": true_com_mag_m,
                "resolution": [384, 384],
                "crop_params": {}
            },
            "frames": []
        }

        raw_buffer =[]
        original_frame_count = 0
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
                    print(f"   [!] Lost tracking after {len(raw_buffer)} frames. Resetting streak...")
                raw_buffer =[]

            if len(raw_buffer) == max_frames:
                print(f"   [OK] Found continuous {max_frames}-frame chunk! Analyzing Kinematics...")
                break
            original_frame_count += 1

        cap.release()

        if len(raw_buffer) == max_frames:
            anchor_idx = 0
            max_pts = 0
            for idx, item in enumerate(raw_buffer):
                if len(item['obj_pts']) > max_pts:
                    max_pts = len(item['obj_pts'])
                    anchor_idx = idx

            poses = [None] * max_frames

            # Solve Anchor
            _, r_anch, t_anch = cv2.solvePnP(raw_buffer[anchor_idx]['obj_pts'], raw_buffer[anchor_idx]['img_pts'], mtx, dist, flags=cv2.SOLVEPNP_SQPNP)
            poses[anchor_idx] = (r_anch, t_anch)

            # Track Forwards
            for i in range(anchor_idx + 1, max_frames):
                _, r, t = cv2.solvePnP(raw_buffer[i]['obj_pts'], raw_buffer[i]['img_pts'], mtx, dist,
                                       rvec=poses[i-1][0].copy(), tvec=poses[i-1][1].copy(),
                                       useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
                poses[i] = (r, t)

            # Track Backwards
            for i in range(anchor_idx - 1, -1, -1):
                _, r, t = cv2.solvePnP(raw_buffer[i]['obj_pts'], raw_buffer[i]['img_pts'], mtx, dist,
                                       rvec=poses[i+1][0].copy(), tvec=poses[i+1][1].copy(),
                                       useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
                poses[i] = (r, t)

            rvecs = np.array([p[0].flatten() for p in poses])
            tvecs = np.array([p[1].flatten() for p in poses])

            # Unwrap rotational vectors
            for i in range(1, max_frames):
                diff = rvecs[i] - rvecs[i-1]
                for j in range(3):
                    while diff[j] > np.pi:
                        rvecs[i][j] -= 2*np.pi
                        diff[j] -= 2*np.pi
                    while diff[j] < -np.pi:
                        rvecs[i][j] += 2*np.pi
                        diff[j] += 2*np.pi

            # =========================================================
            # APPLY GLOBAL EXTREME SMOOTHING (NEWTONIAN REGRESSION)
            # =========================================================
            rvecs, tvecs = apply_newtonian_polyfit(rvecs, tvecs, degree=2)

            continuous_buffer = []
            all_xs, all_ys = [],[]

            for i in range(max_frames):
                rvec = rvecs[i].reshape(3,1)
                tvec = tvecs[i].reshape(3,1)

                box_2d, _ = cv2.projectPoints(box_3d, rvec, tvec, mtx, dist)
                com_2d, _ = cv2.projectPoints(com_3d, rvec, tvec, mtx, dist)

                all_xs.extend(box_2d[:, 0, 0])
                all_ys.extend(box_2d[:, 0, 1])

                continuous_buffer.append({
                    "clean_frame": raw_buffer[i]["frame"],
                    "box_2d": box_2d,
                    "com_u": com_2d[0][0][0],
                    "com_v": com_2d[0][0][1],
                    "rvec": rvec,
                    "tvec": tvec,
                    "original_idx": raw_buffer[i]["original_idx"]
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

            statera_data["metadata"]["crop_params"] = {
                "x1": int(x1), "y1": int(y1), "target_size": int(target_size), "scale": float(scale)
            }

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video = cv2.VideoWriter(out_video_path, fourcc, 60.0, (384, 384))

            for seq_idx, item in enumerate(continuous_buffer):
                clean_frame = item["clean_frame"]
                crop_window = clean_frame[y1:y2, x1:x2]
                final_frame = cv2.resize(crop_window, (384, 384), interpolation=cv2.INTER_AREA)

                final_com_u = (item["com_u"] - x1) * scale
                final_com_v = (item["com_v"] - y1) * scale

                frame_meta = {
                    "sequence_index": seq_idx,
                    "original_video_frame": item["original_idx"],
                    "com_u": float(final_com_u),
                    "com_v": float(final_com_v),
                    "z_depth_meters": float(item["tvec"][2][0]),
                    "com_magnitude_meters": float(true_com_mag_m),
                    "rvec": item["rvec"].flatten().tolist(),
                    "tvec": item["tvec"].flatten().tolist()
                }

                out_video.write(final_frame)
                statera_data["frames"].append(frame_meta)

            out_video.release()
            with open(out_json_path, "w") as f:
                json.dump(statera_data, f, indent=4)
            print(f"   [SUCCESS] Saved sequence to {out_video_path}")
        else:
            print(f"   [ERROR] Skipping output.")

    print("\n[INFO] Batch processing complete!")

if __name__ == "__main__":
    process_directory()