import cv2
import numpy as np
import glob
import os
import warnings
import json
import sys
import random
from PIL import Image, ImageDraw

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
                config[key] = cast_type(val)
            except ValueError:
                print(f"[!] Invalid input. Using default: {default}")
                config[key] = default

        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)

    return config[key]

def generate_checkerboard_pdf(cols, rows, sq_size_cm, dpi):
    squares_x, squares_y = cols + 1, rows + 1
    cm_to_inch = 1 / 2.54
    sq_px = int((sq_size_cm * cm_to_inch) * dpi)
    padding_px = dpi

    width_px = (squares_x * sq_px) + (2 * padding_px)
    height_px = (squares_y * sq_px) + (2 * padding_px)

    img = Image.new('L', (width_px, height_px), color=255)
    draw = ImageDraw.Draw(img)

    for y in range(squares_y):
        for x in range(squares_x):
            if (x + y) % 2 == 1:
                x1, y1 = padding_px + (x * sq_px), padding_px + (y * sq_px)
                draw.rectangle([x1, y1, x1 + sq_px, y1 + sq_px], fill=0)

    pdf_path = "STATERA_Checkerboard.pdf"
    img.save(pdf_path, "PDF", resolution=dpi)
    return pdf_path

def extract_random_spread_frames(video_path, output_dir, num_frames=100):
    """
    Divides the video into equal temporal segments and extracts one random frame
    from each segment. This ensures a wide pose variance across the entire video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"\n[!] ERROR: Cannot open video '{video_path}'. Check the path!")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print("\n[!] ERROR: Video file is empty or corrupted.")
        sys.exit(1)

    target_frames = min(num_frames, total_frames)
    segment_size = total_frames // target_frames

    # Calculate random frame indices spread across the video
    frame_indices =[]
    for i in range(target_frames):
        start_idx = i * segment_size
        end_idx = min((i + 1) * segment_size - 1, total_frames - 1)
        random_frame = random.randint(start_idx, end_idx)
        frame_indices.append(random_frame)

    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[INFO] Extracting {target_frames} randomly spread frames from '{video_path}'...")

    saved_count = 0
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            out_path = os.path.join(output_dir, f"calib_frame_{saved_count:03d}.jpg")
            # Save at 100% quality to avoid AI compression artifacts on the corners
            cv2.imwrite(out_path, frame,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
            saved_count += 1

        # Terminal loading bar
        if saved_count % 10 == 0:
            print(f"   -> Extracted {saved_count}/{target_frames} frames")

    cap.release()
    print(f"[INFO] Extraction complete. Frames saved to '{output_dir}/'.\n")

def calibrate_camera(output_dir: str = "."):
    print("="*70)
    print(" STATERA: Camera Calibration & Frame Extraction")
    print("="*70)

    # Variables
    dpi = get_var('dpi', 'Printer DPI', int, 300)
    calib_input = get_var('calib_input', 'Calibration source (Video file OR Folder path)', str, 'calibration_video.mp4')
    chk_cols = get_var('checkerboard_cols', 'Checkerboard internal corners X', int, 9)
    chk_rows = get_var('checkerboard_rows', 'Checkerboard internal corners Y', int, 6)
    sq_size_cm = get_var('square_size_cm', 'Checkerboard square size in cm', float, 2.5)

    # Generate the physical PDF in case they haven't printed it yet
    pdf_file = generate_checkerboard_pdf(chk_cols, chk_rows, sq_size_cm, dpi)
    print(f"[INFO] Reference checkerboard generated -> '{pdf_file}'")

    working_dir = "calibration_data"

    # Auto-Extraction Logic
    if calib_input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        if not os.path.exists(calib_input):
            print(f"\n[!] ERROR: Could not find video '{calib_input}'. Did you run ffmpeg?")
            sys.exit(1)

        # Extract 100 frames automatically
        extract_random_spread_frames(calib_input, working_dir, num_frames=100)
    else:
        # User provided a folder of images directly
        working_dir = calib_input

    # Load images for calibration
    images = glob.glob(os.path.join(working_dir, '*.jpg')) + glob.glob(os.path.join(working_dir, '*.png'))

    if not images:
        print(f"\n[!] ERROR: No images found in '{working_dir}'.")
        print("    If you haven't recorded your calibration video yet, please do so,")
        print("    normalize it with FFmpeg, and run this script again!")
        sys.exit(0)

    # --- CALIBRATION MATH ---
    checkerboard_size = (chk_cols, chk_rows)
    square_size_m = sq_size_cm / 100.0  # Meters for absolute 3D projection

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((chk_cols * chk_rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:chk_cols, 0:chk_rows].T.reshape(-1, 2)
    objp *= square_size_m

    objpoints, imgpoints = [],[]
    gray_shape = None
    successful_frames = 0

    print(f"[INFO] Computing Optical Geometry from {len(images)} extracted frames...")

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gray_shape is None:
            gray_shape = gray.shape[::-1]

        # Search for the checkerboard
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

        if ret:
            objpoints.append(objp)
            # Mathematical sub-pixel optimization
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            successful_frames += 1

    if successful_frames == 0:
        print("\n[!] ERROR: OpenCV could not find the checkerboard in ANY of the frames.")
        print("    - Check your 'checkerboard_cols' and 'checkerboard_rows' variables.")
        print("    - Ensure the entire board is clearly visible in the video.")
        sys.exit(1)

    print(f"[INFO] Successfully locked geometry in {successful_frames} frames. Calculating Matrix...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)

    # Evaluate Reprojection Integrity
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    total_error = mean_error / len(objpoints)

    print("\n" + "=" * 50)
    print(" CALIBRATION RESULTS")
    print("=" * 50)
    print(f"Reprojection Error: {total_error:.5f} pixels")
    print("-" * 50)

    if total_error > 1.0:
        warnings.warn("CRITICAL WARNING: Reprojection error > 1.0 pixels! Calibration is mathematically compromised. Your video might be too blurry, or the checkerboard size is wrong.", RuntimeWarning)
    else:
        print("[+] Math check passed! Sensor distortion successfully mapped.")

    np.save(os.path.join(output_dir, 'camera_matrix.npy'), mtx)
    np.save(os.path.join(output_dir, 'dist_coeffs.npy'), dist)
    print("[INFO] 'camera_matrix.npy' and 'dist_coeffs.npy' successfully exported.")

if __name__ == "__main__":
    calibrate_camera()