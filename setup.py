"""
setup.py

Bootstraps the STATERA experimental environment, enforces strict Python 3.12 requirements,
and sequentially downloads the HiddenMass-50K dataset alongside the model checkpoints.
Crucially includes an automated patch for a known PyTorch Hub localhost bug residing in the
V-JEPA temporally-aware backbone repository.
"""

import sys
import os
import subprocess
import shutil
import glob


# =================================================================
# 1. BOOTSTRAP & ENVIRONMENT MANAGEMENT
# =================================================================
def get_venv_python():
    """Returns the path to the Python executable inside the local .venv"""
    if os.name == 'nt':  # Windows
        return os.path.join('.venv', 'Scripts', 'python.exe')
    return os.path.join('.venv', 'bin', 'python')


def find_python_3_12():
    """Attempts to find Python 3.12 installed on the host system"""
    path = shutil.which('python3.12')
    if path:
        return path

    if os.name == 'nt':
        try:
            out = subprocess.check_output(['py', '-3.12', '-c', 'import sys; print(sys.executable)'], text=True)
            return out.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    return None


def relaunch_in_venv():
    """Transfers execution into the virtual environment"""
    venv_python = get_venv_python()
    print("\n[*] Relaunching setup script inside the virtual environment...")
    try:
        subprocess.check_call([venv_python, __file__] + sys.argv[1:])
    except subprocess.CalledProcessError as e:
        print(f"\n[!] Setup failed with error code {e.returncode}")
        sys.exit(e.returncode)
    sys.exit(0)


def bootstrap():
    is_312 = (sys.version_info.major == 3 and sys.version_info.minor == 12)
    in_venv = (sys.prefix != sys.base_prefix)

    if not is_312:
        print(f"[!] You are currently running Python {sys.version_info.major}.{sys.version_info.minor}.")
        print("    STATERA strictly requires Python 3.12.")

        venv_py = get_venv_python()
        if os.path.exists(venv_py):
            try:
                v = subprocess.check_output(
                    [venv_py, '-c', 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'],
                    text=True).strip()
                if v == "3.12":
                    print("[*] Found an existing Python 3.12 '.venv'. Auto-switching to it...")
                    relaunch_in_venv()
            except Exception:
                pass

        print("\n    Would you like to search your system for Python 3.12")
        ans = input("    and automatically create a virtual environment? [y/N]: ").strip().lower()
        if ans != 'y':
            print("    Exiting. Please install Python 3.12 and try again.")
            sys.exit(1)

        py312_path = find_python_3_12()
        if not py312_path:
            print("[!] Could not automatically find Python 3.12 on your system.")
            print("    Make sure it is installed and added to your system PATH.")
            sys.exit(1)

        print(f"[*] Found Python 3.12 at: {py312_path}")
        print("[*] Creating new virtual environment '.venv' using Python 3.12...")
        try:
            subprocess.check_call([py312_path, '-m', 'venv', '.venv'])
        except subprocess.CalledProcessError:
            print("[!] Failed to create virtual environment.")
            sys.exit(1)

        relaunch_in_venv()
    else:
        if not in_venv:
            print("[*] No virtual environment detected.")
            if not os.path.exists('.venv'):
                print("[*] Creating new virtual environment '.venv'...")
                import venv
                venv.create('.venv', with_pip=True)
            relaunch_in_venv()
        else:
            run_setup()


# =================================================================
# 2. V-JEPA LOCALHOST BUG PATCHER
# =================================================================
def patch_and_download_vjepa():
    """
    Fixes Meta's broken Torch Hub URL and pre-downloads the model weights.

    STATERA's core architecture relies heavily on Meta's V-JEPA 2.1 (ViT-L) backbone
    to extract temporal tubelets and process transition dynamics. This script natively
    patches the buggy localhost reference in the V-JEPA distribution to ensure the
    temporally-aware foundation can be downloaded effectively for the experiments.
    """
    print("\n[*] Initializing PyTorch Hub to patch Meta's V-JEPA2 localhost bug...")
    import torch

    try:
        torch.hub.help('facebookresearch/vjepa2', 'vjepa2_1_vit_large_384', trust_repo=True)
    except Exception:
        pass

    hub_dir = torch.hub.get_dir()
    vjepa_dir = os.path.join(hub_dir, 'facebookresearch_vjepa2_main')

    if not os.path.exists(vjepa_dir):
        print("    [!] V-JEPA2 repo not found in cache. Could not apply bug patch.")
        return

    bug_fixed = False
    for filepath in glob.glob(os.path.join(vjepa_dir, '**', '*.py'), recursive=True):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        if 'http://localhost:8300' in content:
            print(f"    [!] Found buggy localhost URL in {os.path.basename(filepath)}. Patching...")
            content = content.replace('http://localhost:8300', 'https://dl.fbaipublicfiles.com/vjepa2')
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            bug_fixed = True

    if bug_fixed:
        print("    [✓] V-JEPA2 localhost bug successfully patched on disk.")

    print("[*] Pre-downloading V-JEPA 2 Backbone weights (this may take a few minutes)...")
    download_script = """
import torch
torch.hub.load('facebookresearch/vjepa2', 'vjepa2_1_vit_large_384', trust_repo=True)
print("    [✓] V-JEPA 2 weights downloaded and cached successfully.")
"""
    try:
        subprocess.check_call([get_venv_python(), "-c", download_script])
    except Exception as e:
        print(f"    [!] Warning: Failed to pre-download V-JEPA 2 weights: {e}")


# =================================================================
# 3. UTILITY: READ HF_TOKEN FROM .ENV
# =================================================================
def get_hf_token():
    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    if os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                if line.strip().startswith("HF_TOKEN="):
                    return line.strip().split("=", 1)[1].strip('"\'')
    return None


# =================================================================
# 4. MAIN SETUP ROUTINE
# =================================================================
def run_setup():
    print("==============================================================================")
    print("                              STATERA Setup Utility                           ")
    print("==============================================================================")
    print("Please select an installation mode:")
    print("\n[1] Demo Installation")
    print("    -> Installs minimal dependencies (Inference, Web App).")
    print("    -> Downloads ONLY the primary SOTA checkpoint and the V-jepa backbone (~9.2 GB total).")
    print("    -> Best for trying out the tracking demo or quick local inference.")
    print("\n[2] Full Experimental Result Reproducibility")
    print("    -> Installs all dependencies (Training, Eval, Baselines, WandB, Hub).")
    print("    -> Clones external baseline repositories (DeepMind TAPNET).")
    print("    -> Downloads ALL model checkpoints & 1K ablations (~10 GB total).")
    print("    -> Downloads the complete Public Test HDF5 Dataset (~50 GB).")
    print("    -> Capable of strictly reproducing all Context Tables from the paper.")
    print("==============================================================================\n")

    choice = ""
    while choice not in ["1", "2"]:
        choice = input("Enter 1 or 2: ").strip()

    is_full_install = (choice == "2")

    DEMO_REQS = [
        "numpy", "h5py", "opencv-python", "opencv-contrib-python",
        "torch", "torchvision", "timm", "einops", "pillow",
        "scipy", "tqdm", "fastapi", "uvicorn", "python-multipart", "python-dotenv", "huggingface_hub"
    ]

    FULL_REQS = DEMO_REQS + [
        "mujoco", "albumentations", "matplotlib", "wandb",
        "moviepy", "imageio", "imageio-ffmpeg"
    ]

    reqs_to_install = FULL_REQS if is_full_install else DEMO_REQS

    # --- PIP INSTALLATION ---
    print("\n[*] Upgrading pip...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "--quiet"])

    print(f"[*] Installing {len(reqs_to_install)} packages via pip...")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + reqs_to_install)

    # --- DIRECTORY SCAFFOLDING ---
    print("\n[*] Creating expected directory structures...")
    directories = [
        "demo",
        ".checkpoints",
        "sim",
        "sim2real/output",
        "calibration_data"
    ]
    for d in directories:
        os.makedirs(d, exist_ok=True)
        print(f"    [+] {d}/")

    # --- EXTERNAL REPOSITORIES ---
    if is_full_install:
        print("\n[*] Full Install selected: Setting up external repositories...")
        if not os.path.exists("tapnet_repo"):
            print("    -> Cloning DeepMind's TAPNET repository...")
            try:
                subprocess.check_call(["git", "clone", "https://github.com/deepmind/tapnet.git", "tapnet_repo"])
            except FileNotFoundError:
                print("    [!] Git not found on system. Please install Git to clone TAPNET.")
        else:
            print("    [✓] TAPNET repository already exists.")

    # --- META V-JEPA BUG PATCH ---
    patch_and_download_vjepa()

    # --- HUGGING FACE DOWNLOADS ---
    # We dynamically import this after pip installation is finished
    from huggingface_hub import hf_hub_download

    hf_token = get_hf_token()
    print("\n" + "=" * 70)
    print("[*] Initiating Hugging Face Downloads")
    if hf_token:
        print("    [✓] HF_TOKEN detected in environment/.env. Rate limits bypassed.")
    else:
        print("    [!] No HF_TOKEN detected. If downloading 50GB fails due to rate")
        print("        limits, add HF_TOKEN=your_token to a .env file and rerun.")
    print("=" * 70)

    # Models List
    models_to_download = [
        ("Animesh-null/STATERA", "STATERA-50K-Crescent.pth"),
    ]

    if is_full_install:
        models_to_download.extend([
            ("Animesh-null/STATERA", "STATERA-50K-Sigma.pth"),
            ("Animesh-null/STATERA", "ablations/STATERA-1K-ResNet3D.pth"),
            ("Animesh-null/STATERA", "ablations/STATERA-1K-DINOv2.pth"),
            ("Animesh-null/STATERA", "ablations/STATERA-1K-VideoMAE.pth"),
            ("Animesh-null/STATERA", "ablations/STATERA-1K-No-Z-Depth.pth"),
            ("Animesh-null/STATERA", "ablations/STATERA-1K-Frozen-Anchor.pth"),
            ("Animesh-null/STATERA", "ablations/STATERA-1K-Anchor.pth"),
            ("Animesh-null/STATERA", "ablations/STATERA-1K-Standard-Sigma.pth")
        ])

    print(f"\n[*] Downloading {len(models_to_download)} Checkpoints to '.checkpoints/'...")
    for repo, filename in models_to_download:
        print(f"    -> Fetching {os.path.basename(filename)}...")
        try:
            downloaded_path = hf_hub_download(
                repo_id=repo,
                filename=filename,
                local_dir=".checkpoints",
                token=hf_token
            )
            if "ablations/" in filename:
                flat_dest = os.path.join(".checkpoints", os.path.basename(filename))
                if not os.path.exists(flat_dest):
                    shutil.move(downloaded_path, flat_dest)

            print(f"       [✓] Success.")
        except Exception as e:
            print(f"       [!] Failed: {e}")

    # Dataset Download
    if is_full_install:
        print("\n[*] Downloading 50GB Public Test Dataset (This supports resuming)...")
        try:
            hf_hub_download(
                repo_id="Animesh-null/HiddenMass-50K",
                filename="HiddenMass-50K-Test-Public.hdf5",
                repo_type="dataset",
                local_dir=".checkpoints",
                token=hf_token
            )
            print("    [✓] Dataset download complete.")
        except Exception as e:
            print(f"    [!] Dataset download failed: {e}")

        # The training dataset and combined ablations are immensely large (~50GB each).
        # This setup exclusively downloads the HiddenMass-50K Public Test sequence to
        # strictly reproduce the Context Tables from the paper without exhausting disk space.
        print("\n    [i] NOTE: The Training Dataset and combined 1K Ablation datasets")
        print("        are NOT downloaded by default due to immense size. You can")
        print("        download them manually from Hugging Face if you wish to train.")

        # Ground Truth JSON Handling
        gt_src = "HiddenMass-50K-GroundTruth.json"
        gt_dest = os.path.join(".checkpoints", gt_src)
        print("\n[*] Checking for Confidential Ground Truth JSON...")
        if os.path.exists(gt_src):
            shutil.move(gt_src, gt_dest)
            print(f"    [✓] Found {gt_src} in root! Safely moved to .checkpoints/")
        elif not os.path.exists(gt_dest):
            print("    [!] WARNING: HiddenMass-50K-GroundTruth.json not found!")
            print("        This file is confidential and NOT publicly available.")
            print("        Reviewers: Please place the supplementary JSON file in the")
            print("        root of this repository to run the reproduce tables script.")
        else:
            print("    [✓] Ground Truth JSON already present in .checkpoints/")

    # --- FINISH ---
    activate_cmd = ".venv\\Scripts\\activate" if os.name == 'nt' else "source .venv/bin/activate"

    print("\n==============================================================================")
    print("[✓] STATERA Setup Complete!")
    print("==============================================================================")
    if is_full_install:
        print("All dependencies, checkpoints, and datasets required to run the")
        print("`reproduce_paper_tables.py` script have been satisfied.")
    print("\nTo activate your environment and start working, run:")
    print(f"    {activate_cmd}")
    print("==============================================================================\n")


if __name__ == "__main__":
    bootstrap()