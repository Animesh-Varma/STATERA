import json
import os


def print_ascii_table(data):
    print("\n" + "=" * 110)
    print(f"{'STATERA VANGUARD-V3 FINAL RESULTS (5K VIDEOS, UNIFIED CONTINUOUS FUNNEL)':^110}")
    print("=" * 110)
    print(
        f"| {'Run Name':^28} | {'Heatmap Loss':^14} | {'Mean Error':^12} | {'P95 Error':^12} | {'Kinematic Jitter':^18} |")
    print("-" * 110)

    for key, metrics in sorted(data.items()):
        h_loss = f"{metrics['val_heatmap_loss']:.4f}" if metrics['val_heatmap_loss'] != 999 else "FAILED"
        px_err = f"{metrics['val_pixel_error']:.2f}px" if metrics['val_pixel_error'] != 999 else "FAILED"
        p95_err = f"{metrics['val_p95_error']:.2f}px" if metrics.get('val_p95_error', 999) != 999 else "FAILED"
        jitter = f"{metrics['val_kinematic_jitter']:.3f}" if metrics.get('val_kinematic_jitter',
                                                                         999) != 999 else "FAILED"

        print(f"| {key:<28} | {h_loss:^14} | {px_err:^12} | {p95_err:^12} | {jitter:^18} |")
    print("=" * 110 + "\n")


if __name__ == "__main__":
    metrics_file = "vanguard_v3_metrics.json"

    if not os.path.exists(metrics_file):
        print(f"Error: {metrics_file} not found.")
        exit(1)

    with open(metrics_file, "r") as f:
        metrics_data = json.load(f)

    print_ascii_table(metrics_data)
    print("[✓] Vanguard V3 reporting complete.")