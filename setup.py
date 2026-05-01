import sys
import os
import subprocess
import urllib.request
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

    # --- WRONG PYTHON VERSION DETECTED ---
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

    # --- CORRECT PYTHON VERSION ---
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
    """Fixes Meta's broken Torch Hub URL and pre-downloads the model weights in a clean process"""
    print("\n[*] Initializing PyTorch Hub to patch Meta's V-JEPA2 localhost bug...")
    import torch

    # 1. Force the download/extraction of the V-JEPA2 repository code
    try:
        torch.hub.help('facebookresearch/vjepa2', 'vjepa2_1_vit_large_384', trust_repo=True)
    except Exception:
        pass

    hub_dir = torch.hub.get_dir()
    vjepa_dir = os.path.join(hub_dir, 'facebookresearch_vjepa2_main')

    if not os.path.exists(vjepa_dir):
        print("    [!] V-JEPA2 repo not found in cache. Could not apply bug patch.")
        return

    # 2. Scan and patch all python files containing the internal localhost URL on Disk
    bug_fixed = False
    for filepath in glob.glob(os.path.join(vjepa_dir, '**', '*.py'), recursive=True):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        if 'http://localhost:8300' in content:
            print(f"    [!] Found buggy localhost URL in {os.path.basename(filepath)}. Patching...")
            # Replace local dev URL with Meta's actual public S3 bucket
            content = content.replace('http://localhost:8300', 'https://dl.fbaipublicfiles.com/vjepa2')
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            bug_fixed = True

    if bug_fixed:
        print("    [✓] V-JEPA2 localhost bug successfully patched on disk.")
    else:
        print("    [✓] No buggy localhost URL found (already patched).")

    # 3. Pre-download the weights using a FRESH Subprocess so Python's memory cache is forced to read the patched file
    print("[*] Pre-downloading V-JEPA 2 Backbone weights (this may take a few minutes)...")

    download_script = """
import torch
print("    -> Spawning fresh Python process to load patched V-JEPA2...")
torch.hub.load('facebookresearch/vjepa2', 'vjepa2_1_vit_large_384', trust_repo=True)
print("    [✓] V-JEPA 2 weights downloaded and cached successfully.")
"""
    try:
        venv_python = get_venv_python()
        subprocess.check_call([venv_python, "-c", download_script])
    except Exception as e:
        print(f"    [!] Warning: Failed to pre-download V-JEPA 2 weights: {e}")


# =================================================================
# 3. MAIN SETUP ROUTINE
# =================================================================
def run_setup():
    print("==================================================")
    print("             STATERA Setup Utility                ")
    print("==================================================")
    print("Select an installation mode:")
    print("  [1] Full Installation (Training, Baselines, MuJoCo, WandB, TapNet)")
    print("[2] Demo Installation (Inference, Web App, & core dependencies only)")

    choice = ""
    while choice not in ["1", "2"]:
        choice = input("\nEnter 1 or 2: ").strip()

    is_full_install = (choice == "1")

    DEMO_REQS = [
        "numpy", "h5py", "opencv-python", "opencv-contrib-python",
        "torch", "torchvision", "timm", "einops", "pillow",
        "scipy", "tqdm", "fastapi", "uvicorn", "python-multipart"
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
        "scripts/checkpoints",
        "scripts/sota_50k_crescent_checkpoints",
        "scripts/sota_50k_sigma_checkpoints",
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
    else:
        print("\n[*] Demo Install selected: Skipping TAPNET and training dependencies.")

    # --- META V-JEPA BUG PATCH ---
    patch_and_download_vjepa()

    # --- DOWNLOAD CHECKPOINT ---
    def download_progress(count, block_size, total_size):
        if total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            percent = min(percent, 100)
            sys.stdout.write(f"\r    Downloading... {percent}%")
            sys.stdout.flush()

    checkpoint_url = "https://huggingface.co/Animesh-null/STATERA/resolve/main/STATERA-50K-Crescent.pth"
    checkpoint_dest = os.path.join("demo", "STATERA-50K-Crescent.pth")

    print(f"\n[*] Checking SOTA Checkpoint ({checkpoint_dest})...")
    if not os.path.exists(checkpoint_dest):
        try:
            urllib.request.urlretrieve(checkpoint_url, checkpoint_dest, reporthook=download_progress)
            print("\n    [✓] Download complete.")
        except Exception as e:
            print(f"\n    [!] Failed to download checkpoint: {e}")
    else:
        print("[✓] Checkpoint already exists. Skipping download.")

    # --- FINISH ---
    activate_cmd = ".venv\\Scripts\\activate" if os.name == 'nt' else "source .venv/bin/activate"

    print("\n==================================================")
    print("[✓] STATERA Setup Complete!")
    print("==================================================")
    print(f"To activate your environment and start working, run:")
    print(f"    {activate_cmd}")
    print("==================================================\n")


if __name__ == "__main__":
    bootstrap()