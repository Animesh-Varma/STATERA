"""
3_extract_kinematics_batch.py
Sim-to-Real Dataset Generator.
*FINAL FIX*: Automatically translates MuJoCo (Z-Up) Center of Mass coordinates
into OpenCV (Y-Down) coordinates AND inverts the X-axis to correct for
physical ArUco face mirroring. Includes Scalar Magnitude extraction.
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
                    config[key] = [float(x.strip()) for x in val.split(',')]
                else:
                    config[key] = cast_type(val)
            except Exception:
                print(f"[!] Invalid input. Using default: {default}")
                config[key] = default

        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)

    return config[key]


def process_directory():
    print("="*60)
    print(" STATERA: Kinematics Batch Extraction Setup")
    print("="*60)

    marker_size_cm = get_var('marker_size_cm', 'Target black marker size in cm', float, 10.0)
    box_size_cm = get_var('box_size_cm', 'Box dimensions X, Y, Z in cm', list, [15.0, 15.0, 15.0])

    # User inputs this based on MuJoCo
    mujoco_com_offset_cm = get_var('com_offset_cm', 'MuJoCo Hidden CoM offset (X=Right, Y=Fwd, Z=Up) in cm', list, [4.0, -3.0, 5.0])
    max_frames = get_var('max_frames', 'Number of continuous frames to output per video', int, 16)

    input_dir = get_var('input_dir', 'Path to input video directory', str, 'data')
    output_dir = get_var('output_dir', 'Path to output directory', str, 'output')

    os.makedirs(output_dir, exist_ok=True)
    video_files = glob.glob(os.path.join(input_dir, '*.mp4'))

    if not video_files:
        print(f"[!] No .mp4 files found in '{input_dir}'. Exiting.")
        return

    # ==========================================
    # FINAL COORDINATE FIX (Visualizer Verified)
    # OpenCV Y-Down Mapping AND X-Axis Mirroring
    # ==========================================
    mj_x, mj_y, mj_z = [v / 100.0 for v in mujoco_com_offset_cm]

    cv_x = -mj_x  # <- X-AXIS INVERTED TO MATCH PHYSICAL REALITY
    cv_y = -mj_z  # <- Z-Up becomes Y-Down
    cv_z = mj_y   # <- Y-Fwd becomes Z-Depth

    opencv_com_offset_m = [cv_x, cv_y, cv_z]

    # NEW: Calculate 3D Scalar Magnitude
    true_com_mag_m = math.sqrt(cv_x**2 + cv_y**2 + cv_z**2)

    bx, by, bz = [v / 100.0 for v in box_size_cm]
    hx, hy, hz = bx / 2, by / 2, bz / 2
    hm = (marker_size_cm / 100.0) / 2

    DYNAMIC_MARKERS = {
        0: [[-hm, -hm,  hz], [ hm, -hm,  hz], [ hm,  hm,  hz], [-hm,  hm,  hz]],
        1: [[ hm, -hm, -hz], [-hm, -hm, -hz], [-hm,  hm, -hz], [ hm,  hm, -hz]],
        2: [[-hx, -hm, -hm], [-hx, -hm,  hm], [-hx,  hm,  hm], [-hx,  hm, -hm]],
        3: [[ hx, -hm,  hm], [ hx, -hm, -hm], [ hx,  hm, -hm], [ hx,  hm,  hm]],
        4: [[-hm, -hy, -hm], [ hm, -hy, -hm], [ hm, -hy,  hm], [-hm, -hy,  hm]],
        5: [[-hm,  hy,  hm], [ hm,  hy,  hm], [ hm,  hy, -hm], [-hm,  hy, -hm]]
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
        [-hx, -hy, -hz],[hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy,  hz], [hx, -hy,  hz], [hx, hy,  hz], [-hx, hy,  hz]
    ], dtype=np.float32)

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
                "com_magnitude_m": true_com_mag_m, # Stored at top level
                "resolution": [384, 384]
            },
            "frames": []
        }

        continuous_buffer = []
        original_frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret: break

            corners, ids, _ = detector.detectMarkers(frame)
            detection_success = False

            if ids is not None and len(ids) > 0:
                obj_points, img_points = [], []
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)

                for i, marker_id in enumerate(ids):
                    mid = int(marker_id[0])
                    if mid in DYNAMIC_MARKERS:
                        cv2.fillConvexPoly(mask, np.int32(corners[i][0]), 255)
                        for j in range(4):
                            obj_points.append(DYNAMIC_MARKERS[mid][j])
                            img_points.append(corners[i][0][j])

                if len(img_points) >= 4:
                    _, rvec, tvec = cv2.solvePnP(
                        np.array(obj_points, dtype=np.float32),
                        np.array(img_points, dtype=np.float32),
                        mtx, dist
                    )

                    com_3d = np.array([opencv_com_offset_m], dtype=np.float32)
                    com_2d, _ = cv2.projectPoints(com_3d, rvec, tvec, mtx, dist)
                    com_u, com_v = com_2d[0][0]

                    clean_frame = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
                    box_2d, _ = cv2.projectPoints(box_3d, rvec, tvec, mtx, dist)

                    continuous_buffer.append({
                        "clean_frame": clean_frame,
                        "box_2d": box_2d,
                        "com_u": com_u,
                        "com_v": com_v,
                        "rvec": rvec,
                        "tvec": tvec,
                        "original_idx": original_frame_count
                    })
                    detection_success = True

            if not detection_success:
                if len(continuous_buffer) > 0:
                    print(f"   [!] Lost tracking after {len(continuous_buffer)} frames. Resetting streak...")
                continuous_buffer = []

            if len(continuous_buffer) == max_frames:
                print(f"   [OK] Found continuous {max_frames}-frame chunk! Calculating static crop...")
                break
            original_frame_count += 1

        cap.release()

        if len(continuous_buffer) == max_frames:
            all_xs, all_ys = [], []
            for item in continuous_buffer:
                all_xs.extend(item["box_2d"][:, 0, 0])
                all_ys.extend(item["box_2d"][:, 0, 1])

            global_min_x, global_max_x = int(np.min(all_xs)), int(np.max(all_xs))
            global_min_y, global_max_y = int(np.min(all_ys)), int(np.max(all_ys))

            cx, cy = (global_min_x + global_max_x) // 2, (global_min_y + global_max_y) // 2
            max_dim = max(global_max_x - global_min_x, global_max_y - global_min_y)

            target_size = int(max_dim * 3.0)
            target_size += target_size % 2
            half_size = target_size // 2

            x1, y1 = cx - half_size, cy - half_size
            x2, y2 = cx + half_size, cy + half_size

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video = cv2.VideoWriter(out_video_path, fourcc, 60.0, (384, 384))

            for seq_idx, item in enumerate(continuous_buffer):
                clean_frame = item["clean_frame"]
                crop_canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)

                fx1, fy1 = max(0, x1), max(0, y1)
                fx2, fy2 = min(clean_frame.shape[1], x2), min(clean_frame.shape[0], y2)

                cx1, cy1 = fx1 - x1, fy1 - y1
                cx2, cy2 = cx1 + (fx2 - fx1), cy1 + (fy2 - fy1)

                crop_canvas[cy1:cy2, cx1:cx2] = clean_frame[fy1:fy2, fx1:fx2]
                final_frame = cv2.resize(crop_canvas, (384, 384), interpolation=cv2.INTER_AREA)

                scale = 384.0 / target_size
                final_com_u = (item["com_u"] - x1) * scale
                final_com_v = (item["com_v"] - y1) * scale

                frame_meta = {
                    "sequence_index": seq_idx,
                    "original_video_frame": item["original_idx"],
                    "com_u": float(final_com_u),
                    "com_v": float(final_com_v),
                    "z_depth_meters": float(item["tvec"][2][0]),
                    "com_magnitude_meters": float(true_com_mag_m), # Added to each frame
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