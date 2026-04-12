import torch
import h5py
import numpy as np
import os
import gc
from model import StateraModel
from dataset import StateraDataset


def create_mock_hdf5(filepath):
    print(f"--> Creating mock dataset at {filepath}...")
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('videos', data=np.random.randint(0, 255, (4, 16, 3, 384, 384), dtype=np.uint8))
        f.create_dataset('uv_coords', data=np.random.rand(4, 16, 2) * 384)
        f.create_dataset('z_depths', data=np.random.rand(4, 16))
        f.create_dataset('box_center_uv', data=np.random.rand(4, 16, 2) * 384)


def test_model_config(config_name, kwargs, dataset):
    print(f"Testing [ {config_name} ]...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        model = StateraModel(**kwargs).to(device)
        vids, heatmap, z = dataset[0]
        vids = vids.unsqueeze(0).to(device)

        with torch.no_grad():
            pred_out, pred_z = model(vids)

        if kwargs.get('decoder_type') == 'regression':
            assert pred_out.shape == (1, 16, 2), f"Regression shape mismatch: {pred_out.shape}"
        else:
            assert pred_out.shape == (1, 16, 64, 64), f"Heatmap shape mismatch: {pred_out.shape}"

        assert pred_z.shape == (1, 16, 1), f"Z-Depth shape mismatch: {pred_z.shape}"

        print(f"  ✓ {config_name} Passed Forward Pass.")

    except Exception as e:
        print(f"  ❌ {config_name} FAILED: {e}")
        return False

    finally:
        if 'model' in locals(): del model
        if 'vids' in locals(): del vids
        if 'pred_out' in locals(): del pred_out
        torch.cuda.empty_cache()
        gc.collect()

    return True


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("STATERA PIPELINE PRE-FLIGHT VALIDATION")
    print("=" * 50)

    os.makedirs('../sim', exist_ok=True)
    mock_path = '../sim/mock_statera.hdf5'
    create_mock_hdf5(mock_path)

    # Passing jitter_box=True to test Dataset modifications safely
    dataset = StateraDataset(mock_path, target_type='crescent', jitter_box=True)

    configs = {
        "Run-1-Baseline-MLP": {'decoder_type': 'mlp', 'temporal_mixer': 'conv1d'},
        "Run-2-Deconv-Spatial": {'decoder_type': 'deconv'},
        "Run-6-Transformer": {'decoder_type': 'deconv', 'temporal_mixer': 'transformer'},
        "Run-9-DINOv2": {'decoder_type': 'deconv', 'backbone_type': 'dinov2'},
        "Run-11-Regression": {'decoder_type': 'regression'},
        "Run-12-No-Temporal": {'decoder_type': 'deconv', 'temporal_mixer': 'none'},
        "Run-16-Partial-Finetune": {'decoder_type': 'deconv', 'finetune_blocks': 2},
    }

    all_passed = True
    for name, kwargs in configs.items():
        if not test_model_config(name, kwargs, dataset):
            all_passed = False

    if all_passed:
        print("\n✅ ALL PIPELINES VERIFIED. You are clear to run the 16-run suite.")
    else:
        print("\n🚨 VALIDATION FAILED. Fix the architectures before running the suite.")

    if os.path.exists(mock_path):
        os.remove(mock_path)