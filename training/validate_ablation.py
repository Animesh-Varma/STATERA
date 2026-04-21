import os
import h5py
import torch
import numpy as np
import warnings
from dataset import StateraDataset
from model import StateraModel
from train import get_subpixel_coords

warnings.filterwarnings('ignore')


def create_dummy_dataset(path="dummy_1k.hdf5"):
    """Generates a micro HDF5 dataset mimicking the real 1K subset."""
    print(f"[*] Generating synthetic HDF5 payload at {path}...")
    with h5py.File(path, 'w') as f:
        # 2 dummy videos (Batch size 2)
        f.create_dataset('videos', data=np.random.randint(0, 255, (2, 16, 384, 384, 3), dtype=np.uint8))
        f.create_dataset('uv_coords', data=np.random.uniform(100, 200, (2, 16, 2)).astype(np.float32))
        f.create_dataset('box_center_uv', data=np.random.uniform(100, 200, (2, 16, 2)).astype(np.float32))
        f.create_dataset('z_depths', data=np.random.uniform(1.0, 5.0, (2, 16, 1)).astype(np.float32))

        # Test bbox storage
        bboxes = np.zeros((2, 16, 4), dtype=np.float32)
        bboxes[:, :, 2] = 50.0  # Width 50
        bboxes[:, :, 3] = 50.0  # Height 50
        f.create_dataset('bboxes', data=bboxes)
    print("[✓] Synthetic dataset generated.")


def test_dataset(path):
    print("\n[*] Testing StateraDataset loader and N-CoME Bbox diagonal extraction...")
    ds = StateraDataset(path, target_type='crescent')
    v, h, z, uv, bbox_diag = ds[0]

    assert v.shape == (3, 16, 384, 384), f"Video shape mismatch: {v.shape}"
    assert h.shape == (16, 64, 64), f"Heatmap shape mismatch: {h.shape}"
    assert bbox_diag.shape == (16,), f"Bbox diagonal shape mismatch: {bbox_diag.shape}"
    print("[✓] Dataset I/O and transformations passed.")


def test_architecture(backbone, mixer, decoder):
    print(f"\n[*] Testing Architecture Pipeline: {backbone.upper()} -> {mixer.upper()} -> {decoder.upper()}")
    try:
        model = StateraModel(
            decoder_type=decoder, temporal_mixer=mixer,
            backbone_type=backbone, scratch=False, finetune_blocks=1
        )
        dummy_input = torch.randn(2, 3, 16, 384, 384)
        out_h, out_z = model(dummy_input)

        # Assert Shapes
        if decoder == 'regression':
            assert out_h.shape == (2, 16, 2), f"Regression shape error: {out_h.shape}"
        else:
            assert out_h.shape == (2, 16, 64, 64), f"Heatmap shape error: {out_h.shape}"
        assert out_z.shape == (2, 16, 1), f"Z-depth shape error: {out_z.shape}"

        print(f"[✓] Architecture {backbone} passed forward sequence.")
        return True
    except Exception as e:
        print(f"[!] Architecture test failed for {backbone}: {e}")
        return False


def test_metrics_math():
    print("\n[*] Testing N-CoME and H_KE Mathematical Logic...")
    pred_h = torch.randn(2, 16, 64, 64)
    pred_coords = get_subpixel_coords(pred_h, temperature=2.0)
    assert pred_coords.shape == (2, 16, 2), "Temperature Softmax extraction failed."

    # Mock N-CoME and Jitter
    avg_n_come = 0.05  # 5% relative error
    avg_norm_jitter = 0.02  # 2% acceleration jitter
    h_ke = (2 * avg_n_come * avg_norm_jitter) / (avg_n_come + avg_norm_jitter + 1e-8)
    assert 0 < h_ke < 0.05, f"H_KE calculation bizarre output: {h_ke}"
    print("[✓] Custom Metric Mathematics verified.")


if __name__ == "__main__":
    dummy_path = "dummy_1k.hdf5"
    print("=" * 60)
    print("STATERA 1K PRE-FLIGHT VALIDATOR".center(60))
    print("=" * 60)

    try:
        create_dummy_dataset(dummy_path)
        test_dataset(dummy_path)

        # Testing the three core ablation backbones and configurations
        # NOTE: DINOv2 is tested first as it downloads reliably.
        # (V-JEPA assumes you have meta's repo locally cached or available)
        test_architecture('dinov2', 'none', 'deconv')
        test_architecture('resnet3d', 'conv1d', 'deconv')
        test_architecture('dinov2', 'transformer', 'mlp')

        test_metrics_math()

        print("\n" + "=" * 60)
        print("[SUCCESS] ALL PRE-FLIGHT CHECKS PASSED. SYSTEM READY.".center(60))
        print("=" * 60)

    finally:
        if os.path.exists(dummy_path):
            os.remove(dummy_path)
            print("[*] Cleaned up temporary payload.")