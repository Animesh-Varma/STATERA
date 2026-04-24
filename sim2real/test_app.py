"""
5_review_extracted_kinematics.py
A lightweight OpenCV viewer to manually inspect the extracted HDF5 Ground Truth.
*HDF5 REVISION*:
- Streams directly from .h5 database files.
- Seamlessly scrolls through sequences across multiple files.
- Trajectory Displacement validation and manual/auto database pruning.
"""

import os
import cv2
import glob
import h5py
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


def get_all_sequences(target_dir):
    """Scans all .h5 files and returns a flat list of (filepath, sequence_name) tuples."""
    h5_files = sorted(glob.glob(os.path.join(target_dir, "*_statera.h5")))
    sequence_list =[]

    for h5_file in h5_files:
        try:
            with h5py.File(h5_file, 'r') as f:
                for seq_name in sorted(f.keys()):
                    sequence_list.append((h5_file, seq_name))
        except Exception as e:
            print(f"[!] Could not read {h5_file}: {e}")

    return sequence_list


def delete_empty_h5(h5_file):
    """Deletes the physical .h5 file if all sequences inside it have been pruned."""
    with h5py.File(h5_file, 'r') as f:
        num_keys = len(f.keys())
    if num_keys == 0:
        os.remove(h5_file)
        print(f"  -> Deleted empty database file: {os.path.basename(h5_file)}")


def main():
    print("=" * 60)
    print(" STATERA: HDF5 Kinematics Reviewer & Validator")
    print("=" * 60)

    target_dir = 'output'
    sequence_list = get_all_sequences(target_dir)

    if not sequence_list:
        print(f"[!] No valid sequences found in .h5 databases inside '{target_dir}'.")
        return

    print(f"[INFO] Found {len(sequence_list)} extracted sequences across databases.")
    print("Controls:")
    print("  [Space] - Pause / Play")
    print("  [N]     - Next Sequence")
    print("  [P]     - Previous Sequence")
    print("  [X]     - DELETE current sequence from Database (Negligible movement)")
    print("  [V]     - AUTO-PRUNE all static sequences globally (Displacement < 15px)")
    print("  [Q]     - Quit\n")

    vid_idx = 0

    cv2.namedWindow("Ground Truth Reviewer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Ground Truth Reviewer", 600, 600)

    while vid_idx < len(sequence_list):
        h5_path, seq_name = sequence_list[vid_idx]

        # Load data dynamically from HDF5
        with h5py.File(h5_path, 'r') as f:
            if seq_name not in f:
                # Edge case: was deleted
                vid_idx += 1
                continue

            grp = f[seq_name]
            frames = grp["frames"][:]
            com_u = grp["com_u"][:]
            com_v = grp["com_v"][:]

        trajectory_floats =[np.array([u, v]) for u, v in zip(com_u, com_v)]
        max_displacement = compute_max_displacement(trajectory_floats)
        is_static = max_displacement < 15.0

        frame_idx = 0
        is_playing = True

        while True:
            canvas = frames[frame_idx].copy()

            # Draw historic trajectory line
            for i in range(1, frame_idx + 1):
                pt1 = (int(trajectory_floats[i - 1][0]), int(trajectory_floats[i - 1][1]))
                pt2 = (int(trajectory_floats[i][0]), int(trajectory_floats[i][1]))
                cv2.line(canvas, pt1, pt2, (255, 255, 0), 2, cv2.LINE_AA)

            # Draw current CoM point
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

            # HUD Rendering
            cv2.putText(canvas, f"Seq: {vid_idx + 1}/{len(sequence_list)} | Frame: {frame_idx + 1}/16",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(canvas, f"DB: {os.path.basename(h5_path)} | {seq_name}",
                        (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Movement Validation HUD
            disp_color = (0, 0, 255) if is_static else (0, 255, 0)
            cv2.putText(canvas, f"Max Displacement: {max_displacement:.1f} px", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, disp_color, 2)
            if is_static:
                cv2.putText(canvas, "WARNING: STATIC SEQUENCE (Press 'X' to delete)", (15, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            speed_color = (0, 255, 255)
            accel_color = (0, 0, 255) if accel_val > 5.0 else (255, 165, 0)

            cv2.putText(canvas, f"Velocity: {dist_moved:.1f} px/f", (220, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.6, speed_color, 2)
            cv2.putText(canvas, f"Accel (Jerk): {accel_val:.1f} px/f^2", (180, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.6, accel_color, 2)

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
                # DELETE Current Sequence from HDF5
                with h5py.File(h5_path, 'a') as f:
                    del f[seq_name]
                print(f"\n[DELETED] Removed {seq_name} from {os.path.basename(h5_path)}")

                delete_empty_h5(h5_path)
                sequence_list = get_all_sequences(target_dir) # Refresh list

                # Stay on current vid_idx (which now points to the next sequence)
                if vid_idx >= len(sequence_list):
                    vid_idx = len(sequence_list) - 1
                break

            elif key == ord('v'):
                # AUTO-PRUNE ALL DATABASES
                print("\n[!] Running Auto-Prune globally (Max Displacement < 15px)...")
                prune_count = 0

                # We need a unique list of files to open them securely
                unique_files = list(set([item[0] for item in sequence_list]))

                for f_path in unique_files:
                    with h5py.File(f_path, 'a') as f:
                        for s_name in list(f.keys()):
                            c_u = f[s_name]['com_u'][:]
                            c_v = f[s_name]['com_v'][:]
                            traj = [np.array([u, v]) for u, v in zip(c_u, c_v)]

                            if compute_max_displacement(traj) < 15.0:
                                del f[s_name]
                                prune_count += 1
                                print(f"  -> Pruned: {os.path.basename(f_path)} | {s_name}")

                    delete_empty_h5(f_path)

                print(f"[SUCCESS] Auto-Pruned {prune_count} static sequences globally.")
                sequence_list = get_all_sequences(target_dir)
                vid_idx = 0 # Reset to start
                break

            if is_playing:
                frame_idx += 1
                if frame_idx >= 16:
                    frame_idx = 0

if __name__ == "__main__":
    main()