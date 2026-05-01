
# STATERA
**Hidden Mass Estimation via Zero-Shot Sim-to-Real Kinematics using Frozen Temporal Tubelets**

> *STATERA is a research framework that aims to extract the hidden Center of Mass (CoM) of opaque bodies from raw video using a V-JEPA backbone.*

> [!NOTE]  
> **Provisional README (Pre-Print)**
> 
> STATERA is currently in an early release stage. This README is provisional and will be finalized alongside the upcoming arXiv publication.
> 
> **Asset Availability:** The core pre-trained model checkpoints are currently live on Hugging Face. The massive HiddenMass-50K benchmark dataset and its associated application scripts are still being packaged and will be fully available shortly. 

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Paper](https://img.shields.io/badge/Paper-Coming_Soon_on_arXiv-b31b1b.svg)]()

Standard AI vision models and surface trackers struggle to find the true Center of Mass (CoM) of objects that are asymmetric and opaque. Because the inside is hidden, the problem is mathematically ill-posed for models that only analyze static images. 

**STATERA** (**S**patio-**T**emporal **A**nalysis of **T**ensor **E**mbeddings for **R**igid-body **A**symmetry) solves this by watching how objects move over time. Built on top of Meta's V-JEPA (ViT-L) vision foundation model, STATERA uses a parameter-efficient fine-tuning approach (~2.5M *trainable* parameters) to analyze raw video through pre-trained temporal representations. It learns to infer hidden internal mass directly from real-world physics, momentum, and rotational torque.

Alongside the model, we are releasing the **HiddenMass Benchmark**: 50,000 procedurally generated MuJoCo trajectories (split into 40K train and 10K test) and a rigorously annotated 76-sequence zero-shot real-world physical test set.

---

## Resources & Checkpoints

<div align="left">
    <a href="https://huggingface.co/Animesh-null/STATERA">
        <img src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo-with-title.svg" alt="Models on Hugging Face" height="40">
    </a>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://huggingface.co/datasets/Animesh-null/HiddenMass-50K">
        <img src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo-with-title.svg" alt="Dataset on Hugging Face" height="40">
    </a>
</div>

*Note: The dataset repository is currently a Work In Progress (WIP).*

---

<h3 align="center">Contents</h3>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#interactive-demo">Interactive Demo</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#the-metric-illusion">The Metric Illusion</a>
  <br>
  <a href="#benchmark-results">Benchmark</a> •
  <a href="#model-zoo">Model Zoo</a> •
  <a href="#roadmap--future-work">Roadmap</a> •
  <a href="#technical-stack">Tech Stack</a>
  <br>
  <a href="#quick-start--build-instructions">Build</a> •
  <a href="#contact">Contact</a>
</p>

---

## Features

- **Zero-Shot Sim-to-Real Kinematics:** Train entirely in a MuJoCo simulated vacuum and deploy zero-shot to unconstrained 4K real-world footage. The model tracks true momentum, ignoring heavy real-world textural distractors.
- **Interactive Web UI:** Features a new interactive Material 3 HTML web application (`demo/demo_app.py`) out-of-the-box for evaluating continuous video sequences and visualizing the temporal CoM tracking.
- **Advanced Evaluation Suite:** Includes CLI tools (`tools/evaluate.py`) for computing Shannon Entropy, Expected Spatial Dispersion (ESD), Temporally-Weighted Pixel Error, and the Physics Capture Ratio.
- **Robust Environment Management:** A bulletproof `setup.py` that strictly enforces Python 3.12, auto-generates your virtual environment, and applies a critical aggressive hotfix to bypass Meta's PyTorch caching bug on the `vjepa2` repository.

---

## Interactive Demo

<div align="center">
  <table>
    <tr>
      <td align="center">
        <video src="https://github.com/user-attachments/assets/1aee3f9f-8fe4-4bc1-83d8-79fd2b92d35b" width="400" controls autoplay loop muted playsinline></video>
        <br>
        <i>Evaluating Custom Uploaded Data</i>
      </td>
      <td align="center">
        <video src="https://github.com/user-attachments/assets/7fb9a78e-cbcb-406c-95aa-34598888feb2" width="400" controls autoplay loop muted playsinline></video>
        <br>
        <i>Evaluating Pre-Loaded Demo Sequences</i>
      </td>
    </tr>
  </table>
</div>
---

## How It Works

### **The Problem with Spatial Models**
Standard object-detection pipelines (like DINOv2) predictably point to an object's visual geometric center. While fine for uniformly dense objects, this fails catastrophically for asymmetric payloads (e.g., a hollow ball with a heavy weight glued to one side). Surface point trackers (like TAPIR) also fail due to severe motion blur and self-occlusion when a body tumbles. 

### **The STATERA Pipeline**
Because the Center of Mass is hidden, STATERA treats this as a dynamic *tracking* problem rather than a static *image* problem. 
1. **Temporal Tubelets:** A continuous 16-frame video sequence is compressed into latent spatio-temporal blocks (tubelets) via a partially-frozen V-JEPA 2.1 backbone.
2. **1D Temporal Mixer:** A 1D Convolution extracts the velocity gradients across the sequence, isolating inertial physics from the visual geometry.
3. **Multi-Task Decoder:** The network utilizes a Spatial Preservation Decoder to maintain sub-pixel accuracy. It predicts a 2D continuous probability heatmap while simultaneously predicting a 1D Absolute Z-Depth to ensure the model understands 3D space.
4. **Continuous Extraction:** A Temperature-Scaled Soft-Argmax smoothly extracts a continuous coordinate from the discrete probability grid to eliminate quantization noise (Jitter).

---

## The Metric Illusion & Evaluation Suite

Evaluating kinematics strictly via absolute pixel distance is vulnerable to model "reward-hacking." We empirically identified a fundamental tracking duality we call **Expectation Collapse**. A network will output a massive, highly-uncertain probability blob that naturally collapses near the geometric centroid just to play it safe, artificially gaming the accuracy metrics without actually tracking the hidden mass.

To prevent this, STATERA introduces a strict evaluation suite:
1. **KECS (Kinematic-Euclidean Composite Score):** An L2-norm bound balancing Normalized Center of Mass Error (N-CoME) and Normalized Kinematic Jitter.
2. **ESD (Expected Spatial Dispersion):** A spatial variance metric to measure predictive confidence and penalize blurry, high-entropy "safe" predictions.
3. **Physics Capture Ratio:** Measures true physical disentanglement by calculating the percentage of absolute distance the prediction successfully moves away from the visual centroid toward the true hidden mass.

---

## Benchmark Results

### Zero-Shot Real-World Transfer (N=76 Sequences)
*Evaluated on unconstrained 4K/24FPS physical tumbles. Lower is better for N-CoME, Jitter, KECS, and ESD. Higher is better for Physics Capture.*

| Model Architecture | N-CoME (%) ↓ | Norm Jitter ↓ | KECS ↓ | ESD (px²) ↓ | Physics Capture ↑ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Geometric Centroid** | 64.39% | 0.0167 | 0.6439 | N/A | 0.00% |
| **ResNet3D (Standard CNN)** | 68.05% | 0.0436 | 0.6805 | 392.74 | 36.32% *(Overshoot)* |
| **DINOv2 (Spatial Baseline)**| 27.55% | 0.0415 | 0.2755 | **83.51** | 12.40% |
| **STATERA-50K-Crescent** | 28.19% | 0.0401 | 0.2819 | 141.64 | **34.44%** |
| **STATERA-50K-Sigma** | **21.79%** | **0.0289** | **0.2179** | 98.13 | 18.67% |

> **Analysis:** While `STATERA-50K-Sigma` achieves the mathematical State-of-the-Art (KECS: 0.2179), ESD and Physics Capture metrics reveal this is an illusion driven by statistical expectation collapse. `STATERA-50K-Crescent` serves as the true Kinematic SOTA, achieving a 34.44% physical disentanglement ratio by actively hunting the offset mass. However, its strict precision makes it vulnerable to visual-kinematic aliasing (Bimodal Splits) during high-torque aerial bounces.

---

## Model Zoo

The pre-trained weights are available on Hugging Face. The project utilizes three primary configurations:

- **`STATERA-50K-Crescent.pth` (The Kinematic SOTA):** Trained with phase-aware targets. High physical capture ratio, but prone to bimodal aliasing due to settling-state simulator bias.
- **`STATERA-50K-Sigma.pth` (The Quantitative SOTA):** Trained with phase-agnostic Isotropic Gaussian targets. Highly robust with low jitter, but suffers from expectation collapse (lower physical disentanglement).
- **`STATERA-1K-Anchor.pth`:** Provided for experimental purposes to demonstrate low-data spatial overfitting and temporal starvation.

> **Note on Model Size:** While STATERA is extremely parameter-efficient to *train* (~2.5M trainable parameters), the provided `.pth` checkpoints are **~1.2 GB each**. This is because the checkpoints conveniently bundle the entire frozen ViT-L (Vision Transformer Large) backbone (~300M parameters) so the models can be run out-of-the-box without needing to download and stitch base V-JEPA weights at runtime.

---

## Roadmap & Future Work

- **Dataset Packaging & Evaluation Workflow:** The HiddenMass-50K dataset will be split into 40,000 training sequences and 10,000 testing sequences. The 10K test split will be bundled alongside the real-world physical test set. **Note:** Ground truth coordinates for the test datasets will remain private. A formalized evaluation server/workflow will be set up shortly to process submissions against the hidden ground truth.
- **Compute Constraints:** The models currently provided were strictly constrained to a single RTX 5070 Ti and truncated at Epoch 10. The validation loss indicates the models are still under-trained. Deployment on cluster-scale compute is planned for future iterations.

**The Long-Term Vision:**
Because STATERA’s methodology relies purely on the rotational and inertial mechanics of unobservable masses, the architecture is theoretically scale-invariant. Our future work branches into two primary domains:
- **Astrophysical Hidden Mass:** Adapting STATERA's spatio-temporal tubelets to multi-spectral telescope data to latently deduce macro-scale unobservable mass (e.g., non-baryonic dark matter) strictly from the orbital dynamics of visible matter.
- **Generative Video Physics:** Inverting the STATERA pipeline to map kinematic embeddings directly into Latent Diffusion Models (LDMs), enforcing strict Newtonian momentum and physical realism in generative video synthesis.

---

## Technical Stack

- **Language:** Python 3.12 (Strictly Enforced)
- **Framework:** PyTorch
- **Vision Foundation Model:** Meta V-JEPA 2.1 (ViT-L)
- **Simulation Environment:** MuJoCo
- **Web App / UI:** Material 3 HTML (`demo/demo_app.py`)

---

## Quick Start & Build Instructions

Ensure you have Python 3.12 installed. The repository includes an automated interactive `setup.py` that handles virtual environment creation, Meta V-JEPA dependency patching, and Hugging Face checkpoint downloads.

> **Note:** The setup script is actively tested on **Arch Linux** and **macOS**. Support for other Linux distributions and Windows is currently in beta.

```bash
# Clone the repository
git clone https://github.com/Animesh-Varma/statera-hidden-mass.git
cd statera-hidden-mass

# Run the automated setup script
python setup.py

# Activate the newly created virtual environment
source .venv/bin/activate

# Launch the interactive web demo to test the models out-of-the-box
python demo/demo_app.py
```

For automated evaluation and rendering from the CLI:
```bash
# Render heatmaps for SOTA models
bash scripts/render_sota_heatmaps.sh

# Run the complete evaluation suite
bash scripts/run_all_evals.sh
```

---

## Contact

**Note:** I am a high school student building this project in my spare time. This research represents my entry into bridging foundation models with intuitive physics and Newtonian mechanics. It is an ongoing learning process, so contributors, pull requests, and general advice are always more than welcome!

If you have questions, feedback, or compute resources to help scale the 50K model:
Email: `statera@animeshvarma.dev`
