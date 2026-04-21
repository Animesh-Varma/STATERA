import json
import os

def print_ascii_table(data):
    print("\n" + "=" * 135)
    print(f"{'STATERA 1K ABLATION STUDY RESULTS':^135}")
    print("=" * 135)
    print(f"| {'Run Name':^35} | {'N-CoME (%)':^14} | {'H_KE Metric':^14} | {'Mean Error':^12} | {'P95 Error':^12} | {'Kinematic Jitter':^18} |")
    print("-" * 135)

    for key, metrics in sorted(data.items()):
        n_come = f"{metrics.get('val_n_come', 999)*100:.2f}%" if metrics.get('val_n_come', 999) != 999 else "FAILED"
        h_ke = f"{metrics.get('val_h_ke', 999):.5f}" if metrics.get('val_h_ke', 999) != 999 else "FAILED"
        px_err = f"{metrics['val_pixel_error']:.2f}px" if metrics.get('val_pixel_error', 999) != 999 else "FAILED"
        p95_err = f"{metrics['val_p95_error']:.2f}px" if metrics.get('val_p95_error', 999) != 999 else "FAILED"
        jitter = f"{metrics['val_kinematic_jitter']:.3f}" if metrics.get('val_kinematic_jitter', 999) != 999 else "FAILED"

        print(f"| {key:<35} | {n_come:^14} | {h_ke:^14} | {px_err:^12} | {p95_err:^12} | {jitter:^18} |")
    print("=" * 135 + "\n")

if __name__ == "__main__":
    metrics_file = "1k_ablation_metrics.json"

    if not os.path.exists(metrics_file):
        print(f"Error: {metrics_file} not found.")
        exit(1)

    with open(metrics_file, "r") as f:
        metrics_data = json.load(f)

    print_ascii_table(metrics_data)
    print("[✓] 1K Ablation harmonic reporting complete.")