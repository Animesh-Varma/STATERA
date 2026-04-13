"""
5_review_extracted_kinematics.py
A lightweight, fast OpenCV viewer to manually inspect the extracted ArUco Ground Truth.
*REFINED*: Includes Trajectory Displacement validation and manual/auto dataset pruning.
"""

import os
import cv2
import json
import glob
import numpy as np


def compute_max_displacement(trajectory_floats):
    """Calculates the maximum pixel distance between any two points in the trajectory."""
    max_dist = 0.0
    for i in range(len(trajectory_floats)):
        for j in range(i + 1, len(trajectory_floats)):
            dist = np.linalg.norm(trajectory_floats[i] - trajectory_floats[j])
            if dist > max_dist:
                max_dist = dist
    return max_dist


def main():
    print("=" * 60)
    print(" STATERA: Extracted Kinematics Reviewer & Validator")
    print("=" * 60)

    target_dir = 'output'

    def get_mp4_files():
        return sorted(glob.glob(os.path.join(target_dir, "*_statera.mp4")))

    mp4_files = get_mp4_files()

    if not mp4_files:
        print(f"[!] No _statera.mp4 videos found in '{target_dir}'.")
        return

    print(f"[INFO] Found {len(mp4_files)} extracted sequences.")
    print("Controls:")
    print("  [Space] - Pause / Play")
    print("  [N]     - Next Video")
    print("  [P]     - Previous Video")
    print("  [X]     - DELETE current video & JSON (Negligible movement)")
    print("  [V]     - AUTO-PRUNE all static videos (Displacement < 15px)")
    print("  [Q]     - Quit\n")

    vid_idx = 0

    cv2.namedWindow("Ground Truth Reviewer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Ground Truth Reviewer", 600, 600)

    while vid_idx < len(mp4_files):
        mp4_path = mp4_files[vid_idx]
        base_name = mp4_path.replace("_statera.mp4", "")
        json_path = base_name + "_benchmark.json"

        if not os.path.exists(json_path):
            print(f"[!] Missing JSON for {os.path.basename(mp4_path)}, skipping...")
            vid_idx += 1
            continue

        with open(json_path, 'r') as f:
            data = json.load(f)

        frames_meta = data.get("frames",[])
        if len(frames_meta) != 16:
            vid_idx += 1
            continue

        cap = cv2.VideoCapture(mp4_path)
        frames =[]
        for _ in range(16):
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
        cap.release()

        if len(frames) != 16:
            vid_idx += 1
            continue

        trajectory_floats =[]
        for f_meta in frames_meta:
            u, v = float(f_meta["com_u"]), float(f_meta["com_v"])
            trajectory_floats.append(np.array([u, v]))

        max_displacement = compute_max_displacement(trajectory_floats)
        is_static = max_displacement < 15.0

        frame_idx = 0
        is_playing = True

        while True:
            canvas = frames[frame_idx].copy()

            for i in range(1, frame_idx + 1):
                pt1 = (int(trajectory_floats[i - 1][0]), int(trajectory_floats[i - 1][1]))
                pt2 = (int(trajectory_floats[i][0]), int(trajectory_floats[i][1]))
                cv2.line(canvas, pt1, pt2, (255, 255, 0), 2, cv2.LINE_AA)

            curr_pt = trajectory_floats[frame_idx]
            curr_u, curr_v = int(curr_pt[0]), int(curr_pt[1])
            cv2.circle(canvas, (curr_u, curr_v), 6, (255, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, (curr_u, curr_v), 2, (255, 255, 255), -1, cv2.LINE_AA)

            dist_moved = 0.0
            accel_val = 0.0

            if frame_idx > 0:
                v_curr = trajectory_floats[frame_idx] - trajectory_floats[frame_idx - 1]
                dist_moved = np.linalg.norm(v_curr)

            if frame_idx > 1:
                v_curr = trajectory_floats[frame_idx] - trajectory_floats[frame_idx - 1]
                v_prev = trajectory_floats[frame_idx - 1] - trajectory_floats[frame_idx - 2]
                accel_vector = v_curr - v_prev
                accel_val = np.linalg.norm(accel_vector)

            cv2.putText(canvas, f"Seq: {vid_idx + 1}/{len(mp4_files)} | Frame: {frame_idx + 1}/16",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(canvas, os.path.basename(mp4_path),
                        (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Movement Validation HUD
            disp_color = (0, 0, 255) if is_static else (0, 255, 0)
            cv2.putText(canvas, f"Max Displacement: {max_displacement:.1f} px", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, disp_color, 2)
            if is_static:
                cv2.putText(canvas, "WARNING: STATIC VIDEO (Press 'X' to delete)", (15, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            speed_color = (0, 255, 255)
            accel_color = (0, 0, 255) if accel_val > 5.0 else (255, 165, 0)

            cv2.putText(canvas, f"Velocity: {dist_moved:.1f} px/f", (220, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.6, speed_color, 2)
            cv2.putText(canvas, f"Accel (Jitter): {accel_val:.1f} px/f^2", (180, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.6, accel_color, 2)

            status = "PLAYING" if is_playing else "PAUSED"
            cv2.putText(canvas, status, (15, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            cv2.imshow("Ground Truth Reviewer", canvas)

            delay = 60 if is_playing else 0
            key = cv2.waitKey(delay) & 0xFF

            if key == ord('q'):
                cv2.destroyAllWindows()
                return
            elif key == ord('n'):
                vid_idx += 1
                break
            elif key == ord('p'):
                vid_idx = max(0, vid_idx - 1)
                break
            elif key == ord('r'):
                frame_idx = 0
                is_playing = True
            elif key == ord(' '):
                is_playing = not is_playing
            elif key == ord('x'):
                # DELETE Current Video
                os.remove(mp4_path)
                os.remove(json_path)
                print(f"\n[DELETED] Removed static sequence: {os.path.basename(mp4_path)}")
                mp4_files = get_mp4_files()  # Refresh list
                # Keep vid_idx the same so it naturally advances to the new video at this index
                break
            elif key == ord('v'):
                # AUTO-PRUNE ALL
                print("\n[!] Running Auto-Prune for all videos with Max Displacement < 15px...")
                prune_count = 0
                for f_path in mp4_files:
                    j_path = f_path.replace("_statera.mp4", "_benchmark.json")
                    if not os.path.exists(j_path): continue

                    with open(j_path, 'r') as jf:
                        j_data = json.load(jf)

                    traj = [np.array([float(f["com_u"]), float(f["com_v"])]) for f in j_data.get("frames",[])]
                    if len(traj) == 16 and compute_max_displacement(traj) < 15.0:
                        os.remove(f_path)
                        os.remove(j_path)
                        prune_count += 1
                        print(f"  -> Pruned: {os.path.basename(f_path)}")

                print(f"[SUCCESS] Auto-Pruned {prune_count} static videos.")
                mp4_files = get_mp4_files()
                vid_idx = 0 # Reset to start
                break

            if is_playing:
                frame_idx += 1
                if frame_idx >= 16:
                    frame_idx = 0


if __name__ == "__main__":
    main()