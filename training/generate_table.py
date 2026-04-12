import json
import os
import matplotlib.pyplot as plt


def print_ascii_table(data):
    print("\n" + "=" * 85)
    print(f"{'STATERA 1,000-VIDEO ABLATION STUDY RESULTS':^85}")
    print("=" * 85)
    print(f"| {'Run Name':^30} | {'Val Heatmap Loss':^18} | {'Z-Depth Loss':^12} | {'Pixel Error':^11} |")
    print("-" * 85)

    for key, metrics in sorted(data.items()):
        h_loss = f"{metrics['val_heatmap_loss']:.4f}" if metrics['val_heatmap_loss'] != 999 else "FAILED"
        z_loss = f"{metrics['val_z_loss']:.4f}" if metrics['val_z_loss'] != 999 else "FAILED"
        px_err = f"{metrics['val_pixel_error']:.2f}px" if metrics['val_pixel_error'] != 999 else "FAILED"

        if "Run-4" in key:
            key = f"⭐ {key}"

        print(f"| {key:<30} | {h_loss:^18} | {z_loss:^12} | {px_err:^11} |")

    print("=" * 85 + "\n")


def generate_mega_plots(data):
    print("Generating comprehensive Matplotlib Mega-Graphs...")
    runs = sorted(data.keys())
    valid_runs = [r for r in runs if data[r]['val_pixel_error'] != 999.0]

    # ---------------------------------------------------------
    # PLOT 1: PIXEL ERROR
    # ---------------------------------------------------------
    errors = [data[r]['val_pixel_error'] for r in valid_runs]

    plt.figure(figsize=(14, 8))
    bars = plt.barh(valid_runs, errors, color='skyblue', edgecolor='black')

    for bar, run_name in zip(bars, valid_runs):
        if 'Run-4' in run_name:
            bar.set_color('gold')
        elif 'Baseline' in run_name:
            bar.set_color('salmon')

    plt.xlabel('Final Validation Pixel Error (384x384 Native Video)')
    plt.title('STATERA: Tracking Accuracy Comparison (Lower is Better)')
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    for index, value in enumerate(errors):
        plt.text(value + 0.1, index, f'{value:.2f}px', va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig("STATERA_Pixel_Error_Comparison.png", dpi=300)
    plt.close()
    print("✓ Pixel Error Graph saved to 'STATERA_Pixel_Error_Comparison.png'")

    # ---------------------------------------------------------
    # PLOT 2: VAL HEATMAP LOSS
    # ---------------------------------------------------------
    losses = [data[r]['val_heatmap_loss'] for r in valid_runs]

    plt.figure(figsize=(14, 8))
    bars = plt.barh(valid_runs, losses, color='lightgreen', edgecolor='black')

    for bar, run_name in zip(bars, valid_runs):
        if 'Run-4' in run_name:
            bar.set_color('gold')
        elif 'Baseline' in run_name:
            bar.set_color('salmon')

    plt.xlabel('Final Validation Heatmap Loss (Weighted BCE)')
    plt.title('STATERA: Heatmap Convergence Comparison (Lower is Better)')
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    for index, value in enumerate(losses):
        plt.text(value + 0.01, index, f'{value:.4f}', va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig("STATERA_Heatmap_Loss_Comparison.png", dpi=300)
    plt.close()
    print("✓ Heatmap Loss Graph saved to 'STATERA_Heatmap_Loss_Comparison.png'")


if __name__ == "__main__":
    if not os.path.exists("run_metrics.json"):
        print("Error: run_metrics.json not found.")
        exit(1)

    with open("run_metrics.json", "r") as f:
        metrics_data = json.load(f)

    print_ascii_table(metrics_data)
    generate_mega_plots(metrics_data)