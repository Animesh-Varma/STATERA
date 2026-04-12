import json
import os
import matplotlib.pyplot as plt


def print_ascii_table(data):
    print("\n" + "=" * 110)
    print(f"{'STATERA HERO-PHASE FINAL RESULTS (50 EPOCHS)':^110}")
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

        if "Unchained" in key:
            key = f"⛓️‍💥{key}"

        print(f"| {key:<28} | {h_loss:^14} | {px_err:^12} | {p95_err:^12} | {jitter:^18} |")
    print("=" * 110 + "\n")


def plot_metric(valid_runs, data, metric_key, title, xlabel, filename, highlight='Unchained'):
    values = [data[r][metric_key] for r in valid_runs]
    plt.figure(figsize=(12, 5))
    bars = plt.barh(valid_runs, values, color='skyblue', edgecolor='black')

    for bar, run_name in zip(bars, valid_runs):
        if highlight in run_name:
            bar.set_color('gold')
        elif 'Baseline' in run_name:
            bar.set_color('salmon')

    plt.xlabel(xlabel)
    plt.title(title)
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    for index, value in enumerate(values):
        label = f'{value:.2f}px' if 'error' in metric_key else f'{value:.4f}'
        plt.text(value + (max(values) * 0.01), index, label, va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def generate_hero_plots(data):
    print("Generating Matplotlib Hero-Graphs...")
    runs = sorted(data.keys())
    valid_runs = [r for r in runs if data[r]['val_pixel_error'] != 999.0]

    plot_metric(valid_runs, data, 'val_pixel_error', 'STATERA: Mean Pixel Error (Lower is Better)',
                'Mean Euclidean Distance (Pixels)', 'STATERA_HERO_Mean_Error.png')
    plot_metric(valid_runs, data, 'val_p95_error', 'STATERA: P95 Outlier Error (Lower is Better)',
                '95th Percentile Error (Pixels)', 'STATERA_HERO_P95_Error.png')
    plot_metric(valid_runs, data, 'val_kinematic_jitter', 'STATERA: Kinematic Jitter (Lower is Better)',
                'Mean Acceleration Deviation (px/frame^2)', 'STATERA_HERO_Jitter.png')
    plot_metric(valid_runs, data, 'val_heatmap_loss', 'STATERA: Heatmap Convergence', 'Validation Weighted BCE',
                'STATERA_HERO_Heatmap_Loss.png')

    print("✓ All 4 Hero Graphs saved successfully!")


if __name__ == "__main__":
    if not os.path.exists("hero_metrics.json"):
        print("Error: hero_metrics.json not found.")
        exit(1)

    with open("hero_metrics.json", "r") as f:
        metrics_data = json.load(f)

    print_ascii_table(metrics_data)
    generate_hero_plots(metrics_data)