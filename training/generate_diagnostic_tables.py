import json
import os
import matplotlib.pyplot as plt


def generate_report(metrics_file, output_img):
    if not os.path.exists(metrics_file):
        print(f"Error: {metrics_file} not found.")
        return

    with open(metrics_file, 'r') as f:
        data = json.load(f)

    print("\n" + "=" * 100)
    print(f"{'STATERA DIAGNOSTIC & VERIFICATION RESULTS':^100}")
    print("=" * 100)
    print(f"| {'Run Name':^35} | {'Mean Error':^12} | {'P95 Error':^12} | {'Kinematic Jitter':^18} |")
    print("-" * 100)

    for key, m in sorted(data.items()):
        px = f"{m['val_pixel_error']:.2f}px" if m['val_pixel_error'] != 999 else "FAIL"
        p95 = f"{m['val_p95_error']:.2f}px" if m['val_p95_error'] != 999 else "FAIL"
        jit = f"{m['val_kinematic_jitter']:.4f}" if m['val_kinematic_jitter'] != 999 else "FAIL"
        print(f"| {key:<35} | {px:^12} | {p95:^12} | {jit:^18} |")
    print("=" * 100 + "\n")

    # Generate Comparative Bar Chart
    runs = sorted(data.keys())
    jitters = [data[r]['val_kinematic_jitter'] for r in runs if data[r]['val_kinematic_jitter'] != 999]
    errors = [data[r]['val_pixel_error'] for r in runs if data[r]['val_pixel_error'] != 999]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.bar(runs, jitters, color='salmon', edgecolor='black')
    ax1.set_title('Kinematic Jitter (Lower is Better)')
    ax1.set_ylabel('Acceleration Deviation')
    ax1.tick_params(axis='x', rotation=15)

    ax2.bar(runs, errors, color='skyblue', edgecolor='black')
    ax2.set_title('Mean Pixel Error (Lower is Better)')
    ax2.set_ylabel('Pixels')
    ax2.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    print(f"✓ Diagnostic graph saved as {output_img}")


if __name__ == "__main__":
    generate_report("diagnostic_metrics.json", "STATERA_DIAGNOSTIC_REPORT.png")