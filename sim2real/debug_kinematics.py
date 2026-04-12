"""
3b_debug_kinematics_batch.py
Visual Debugger for Sim-to-Real Kinematics (Batch Version).
Processes videos in the 'data/' directory, disables inpainting,
and draws a bright green dot at the calculated Center of Mass.
Now uses a 3.0x WIDE STATIC tracking and prints Z-Depth (meters) on the frame.
"""

import cv2
import numpy as np
import json
import os
import glob

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

def run_debug_batch():
    print("="*70)
    print(" STATERA: Kinematics Visual Debugger (Wide Static Perspective)")
    print("="*70)
    print("This script draws a GREEN DOT at the Center of Mass and imprints the")
    print("calculated Z-DEPTH (meters) on the frame. Check 'debug/' for output.")
    print("-" * 70)

    input_dir = 'data'
    output_dir = 'debug'
    os.makedirs(output_dir, exist_ok=True)

    video_files = glob.glob(os.path.join(input_dir, '*.mp4'))
    if not video_files:
        print(f"[!] No .mp4 files found in '{input_dir}'. Exiting.")
        return

    # Load the physical variables
    marker_size_cm = get_var('marker_size_cm', 'Target black marker size in cm', float, 10.0)
    box_size_cm = get_var('box_size_cm', 'Box dimensions X, Y, Z in cm', list, [15.0, 15.0, 15.0])
    com_offset_cm = get_var('com_offset_cm', 'Hidden CoM offset X, Y, Z in cm', list, [4.0, -3.0, 5.0])
    max_frames = get_var('max_frames', 'Number of continuous frames to output per video', int, 16)

    bx, by, bz = [v / 100.0 for v in box_size_cm]
    com_offset_m = [v / 100.0 for v in com_offset_cm]

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

    # Load Camera Intrinsics
    print("\n[INFO] Loading Intrinsic Parameters...")
    try:
        mtx = np.load('camera_matrix.npy')
        dist = np.load('dist_coeffs.npy')
    except FileNotFoundError:
        print("[!] camera_matrix.npy or dist_coeffs.npy missing. Run calibration first!")
        return

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    box_3d = np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy,  hz], [hx, -hy,  hz], [hx, hy,  hz], [-hx, hy,  hz]
    ], dtype=np.float32)

    for v_idx, video_path in enumerate(video_files):
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"\n[->] Debugging Video {v_idx + 1}/{len(video_files)}: {base_name}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"   [!] Cannot open {video_path}. Skipping.")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
        out_video_path = os.path.join(output_dir, f"{base_name}_debug.mp4")

        continuous_buffer = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            corners, ids, _ = detector.detectMarkers(frame)
            detection_success = False

            if ids is not None and len(ids) > 0:
                obj_points, img_points = [], []

                for i, marker_id in enumerate(ids):
                    mid = int(marker_id[0])
                    if mid in DYNAMIC_MARKERS:
                        for j in range(4):
                            obj_points.append(DYNAMIC_MARKERS[mid][j])
                            img_points.append(corners[i][0][j])

                if len(img_points) >= 4:
                    _, rvec, tvec = cv2.solvePnP(
                        np.array(obj_points, dtype=np.float32),
                        np.array(img_points, dtype=np.float32),
                        mtx, dist
                    )

                    # ---------------------------------------------------------
                    # KINEMATICS MATH
                    # ---------------------------------------------------------
                    com_3d = np.array([com_offset_m], dtype=np.float32)
                    com_2d, _ = cv2.projectPoints(com_3d, rvec, tvec, mtx, dist)
                    com_u, com_v = com_2d[0][0]

                    box_2d, _ = cv2.projectPoints(box_3d, rvec, tvec, mtx, dist)

                    z_depth = float(tvec[2][0])

                    continuous_buffer.append({
                        "frame": frame,
                        "box_2d": box_2d,
                        "com_u": com_u,
                        "com_v": com_v,
                        "z_depth": z_depth
                    })
                    detection_success = True

            # Reset streak if tracking fails
            if not detection_success:
                continuous_buffer = []

            if len(continuous_buffer) == max_frames:
                break

        cap.release()

        if len(continuous_buffer) == max_frames:
            # 1. Determine global bounds
            all_xs, all_ys = [], []
            for item in continuous_buffer:
                all_xs.extend(item["box_2d"][:, 0, 0])
                all_ys.extend(item["box_2d"][:, 0, 1])

            global_min_x, global_max_x = int(np.min(all_xs)), int(np.max(all_xs))
            global_min_y, global_max_y = int(np.min(all_ys)), int(np.max(all_ys))

            cx, cy = (global_min_x + global_max_x) // 2, (global_min_y + global_max_y) // 2
            max_dim = max(global_max_x - global_min_x, global_max_y - global_min_y)

            # Wide 3.0x Crop
            target_size = int(max_dim * 3.0)
            target_size += target_size % 2
            half_size = target_size // 2

            x1, y1 = cx - half_size, cy - half_size

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video = cv2.VideoWriter(out_video_path, fourcc, fps, (384, 384))

            # 2. Render sequence with depth overlay
            for item in continuous_buffer:
                raw_frame = item["frame"]
                crop_canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)

                fx1, fy1 = max(0, x1), max(0, y1)
                fx2, fy2 = min(raw_frame.shape[1], x1 + target_size), min(raw_frame.shape[0], y1 + target_size)

                cx1, cy1 = max(0, -x1), max(0, -y1)
                cw, ch = (fx2 - fx1), (fy2 - fy1)

                crop_canvas[cy1:cy1+ch, cx1:cx1+cw] = raw_frame[fy1:fy2, fx1:fx2]
                final_frame = cv2.resize(crop_canvas, (384, 384), interpolation=cv2.INTER_AREA)

                scale = 384.0 / target_size
                final_com_u = (item["com_u"] - x1) * scale
                final_com_v = (item["com_v"] - y1) * scale

                # DRAW GREEN DOT (CoM)
                cv2.circle(final_frame, (int(final_com_u), int(final_com_v)), 5, (0, 255, 0), -1)

                # IMPRINT Z-DEPTH (METERS)
                depth_text = f"Z: {item['z_depth']:.4f}m"
                cv2.putText(final_frame, depth_text, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

                out_video.write(final_frame)

            out_video.release()
            print(f"   [SUCCESS] Saved debug video to {out_video_path}")
        else:
            print(f"   [ERROR] No {max_frames} frame sequence found in {base_name}.")

if __name__ == "__main__":
    run_debug_batch()