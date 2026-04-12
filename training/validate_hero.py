import torch
import h5py
import numpy as np
import os
import gc
from model import StateraModel
from dataset import StateraDataset


def create_mock_hdf5(filepath):
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('videos', data=np.random.randint(0, 255, (2, 16, 3, 384, 384), dtype=np.uint8))
        f.create_dataset('uv_coords', data=np.random.rand(2, 16, 2) * 384)
        f.create_dataset('z_depths', data=np.random.rand(2, 16))
        f.create_dataset('box_center_uv', data=np.random.rand(2, 16, 2) * 384)


def test_model_config(config_name, kwargs, dataset):
    print(f"Testing [ {config_name} ]...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model = StateraModel(**kwargs).to(device)
        # UNPACKING 4 VARIABLES
        vids, heatmap, z, true_uv = dataset[0]
        vids = vids.unsqueeze(0).to(device)

        with torch.no_grad():
            pred_out, pred_z = model(vids)

        print(f"  ✓ {config_name} Passed Forward Pass.")
    except Exception as e:
        print(f"{config_name} FAILED: {e}")
        return False
    finally:
        if 'model' in locals(): del model
        torch.cuda.empty_cache()
        gc.collect()
    return True


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("STATERA HERO-PHASE PRE-FLIGHT VALIDATION")
    print("=" * 50)
    os.makedirs('../sim', exist_ok=True)
    mock_path = '../sim/mock_hero.hdf5'
    create_mock_hdf5(mock_path)

    dataset = StateraDataset(mock_path, target_type='crescent')

    configs = {
        "Hero-1-Baseline": {'decoder_type': 'deconv'},
        "Hero-4-Unchained": {'decoder_type': 'deconv', 'finetune_blocks': 2},
    }

    all_passed = True
    for name, kwargs in configs.items():
        if not test_model_config(name, kwargs, dataset):
            all_passed = False

    if all_passed:
        print("\nALL HERO PIPELINES VERIFIED. Clear for 50-Epoch Runs.")
    else:
        print("\nVALIDATION FAILED.")

    if os.path.exists(mock_path):
        os.remove(mock_path)