"""
1_generate_markers.py
Generates ArUco markers with physical sizing driven by TUI inputs.
Compiles them automatically into a perfectly scaled, printable A4 PDF.
"""

import cv2
import os
import numpy as np
import json
from PIL import Image

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

def generate_markers(output_dir: str = "markers"):
    os.makedirs(output_dir, exist_ok=True)

    print("="*60)
    print(" STATERA: Printable A4 Marker Generation Setup")
    print(" Delete 'variables.txt' to reset these parameters anytime.")
    print("="*60)

    dpi = get_var('dpi', 'Printer DPI', int, 300)
    marker_size_cm = get_var('marker_size_cm', 'Target black marker size in cm', float, 10.0)
    border_size_cm = get_var('border_size_cm', 'White border padding size in cm', float, 2.54)

    # Mathematical conversion: cm to pixels based on DPI
    CM_TO_INCH = 1 / 2.54
    marker_size_px = int((marker_size_cm * CM_TO_INCH) * dpi)
    border_size_px = int((border_size_cm * CM_TO_INCH) * dpi)
    total_marker_px = marker_size_px + (2 * border_size_px)

    # A4 Paper Dimensions
    a4_w_cm, a4_h_cm = 21.0, 29.7
    a4_w_px = int((a4_w_cm * CM_TO_INCH) * dpi)
    a4_h_px = int((a4_h_cm * CM_TO_INCH) * dpi)

    print(f"\n[INFO] Calculating layout for A4 Paper ({a4_w_cm}x{a4_h_cm} cm) at {dpi} DPI...")
    print(f"       Paper Resolution: {a4_w_px} x {a4_h_px} px")
    print(f"       Marker Resolution (with border): {total_marker_px} x {total_marker_px} px")

    # Calculate Maximum Grid Layout for A4
    cols = a4_w_px // total_marker_px
    rows = a4_h_px // total_marker_px

    if cols == 0 or rows == 0:
        print("[!] ERROR: The requested marker size is too large to fit on an A4 paper!")
        print("    Please reduce the marker size or border padding.")
        return

    markers_per_page = cols * rows
    print(f"       Layout: {cols} columns x {rows} rows ({markers_per_page} markers per page)")

    # Calculate margins to strictly center the grid on the paper
    margin_x = (a4_w_px - (cols * total_marker_px)) // 2
    margin_y = (a4_h_px - (rows * total_marker_px)) // 2

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    pdf_pages =[]
    current_page_canvas = np.full((a4_h_px, a4_w_px), 255, dtype=np.uint8) # White A4 canvas

    marker_id = 0
    total_markers_needed = 6

    while marker_id < total_markers_needed:
        # Reset canvas for new pages
        if marker_id > 0 and marker_id % markers_per_page == 0:
            pdf_pages.append(Image.fromarray(current_page_canvas))
            current_page_canvas = np.full((a4_h_px, a4_w_px), 255, dtype=np.uint8)

        # Determine position in the grid
        idx_on_page = marker_id % markers_per_page
        current_col = idx_on_page % cols
        current_row = idx_on_page // cols

        # Exact pixel offsets for projection onto the canvas
        start_x = margin_x + (current_col * total_marker_px)
        start_y = margin_y + (current_row * total_marker_px)

        # Generate the raw OpenCV grayscale marker
        marker_image = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size_px)

        # Wrap it in the measured white padding
        bordered_marker = cv2.copyMakeBorder(
            marker_image,
            top=border_size_px, bottom=border_size_px,
            left=border_size_px, right=border_size_px,
            borderType=cv2.BORDER_CONSTANT,
            value=255 # White
        )

        # Implant the marker into the A4 mathematical coordinate space
        current_page_canvas[
            start_y : start_y + total_marker_px,
            start_x : start_x + total_marker_px
        ] = bordered_marker

        marker_id += 1

    # Append the last rendered page
    pdf_pages.append(Image.fromarray(current_page_canvas))

    # Compile entirely into a PDF via Pillow
    pdf_output_path = os.path.join(output_dir, "STATERA_Markers_A4.pdf")
    if pdf_pages:
        pdf_pages[0].save(
            pdf_output_path,
            "PDF",
            resolution=dpi,
            save_all=True,
            append_images=pdf_pages[1:]
        )

    print(f"\n[INFO] Success! Complete printable A4 layout saved to -> '{pdf_output_path}'")

if __name__ == "__main__":
    generate_markers()