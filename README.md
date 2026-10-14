# IIT Mandi Hackathon — Kuro Siwo SAR Flood Mapping

Automated flood inundation mapping using multi-temporal Sentinel-1 Synthetic Aperture Radar (SAR) imagery and deep convolutional segmentation networks (U-Net with ResNet-50 encoder).

---

## 📊 Model Highlights & Performance (At a Glance)

| Parameter / Metric | Specification / Result |
| :--- | :--- |
| **Model Architecture** | **U-Net** with **ResNet-50** backbone (ImageNet pre-trained) |
| **Input Channels (6 SAR channels)** | Multi-temporal: `sec1_vv`, `sec1_vh`, `sec2_vv`, `sec2_vh`, `flood_vv`, `flood_vh` |
| **Target Classes (3)** | `0: no_water`, `1: permanent_water`, `2: flood` |
| **Model Weights File** | `outputs_e2e/best_unet_resnet50.pth` (Stored via **Git LFS**) |
| **Best Checkpoint Epoch** | Epoch 2 (selected via fixed validation argmax flood IoU) |
| **Calibrated Flood Threshold** | **0.30** (calibrated on validation split after checkpoint selection) |
| **Validation Flood IoU (Threshold 0.3)** | **77.74%** (`0.7774`) |
| **Validation Flood F1-Score** | **87.47%** (`0.8747`) |
| **Validation Flood Precision / Recall** | Precision: **85.96%** (`0.8596`) \| Recall: **89.04%** (`0.8904`) |
| **Kuro Siwo Test Split IoU** | **50.90%** (`0.5090`) (Threshold 0.3) / **53.07%** (`0.5307`) (Argmax) |
| **Kuro Siwo Test Split F1-Score** | **67.46%** (`0.6746`) (Threshold 0.3) / **69.34%** (`0.6934`) (Argmax) |
| **Kuro Siwo Test Split Precision / Recall** | Precision: **61.70%** / **68.09%** \| Recall: **74.42%** / **70.64%** |

---

## 📁 Repository Structure

```text
├── .gitattributes                          # Git LFS tracking configuration (*.pth, *.pt)
├── .gitignore                              # Ignores large raw dataset shards and caches
├── README.md                               # Project documentation & benchmark metrics
├── flood_mapping_training_corrected.ipynb  # Primary corrected training & evaluation notebook
├── dataset/
│   └── flood_mapping_training_corrected.ipynb  # Notebook copy in dataset directory
└── outputs_e2e/
    ├── best_unet_resnet50.pth              # Best trained model state dictionary (Git LFS)
    ├── model_metadata.json                 # Model hyperparams, channels, normalization & thresholds
    ├── training_history.json               # Full per-epoch loss, IoU, learning rate progression
    ├── kuro_siwo_test_report.json          # Complete evaluation metrics on Kuro Siwo test set
    └── results.json                        # Summary test performance metrics
```

---

## 🛠️ Pipeline Details & Key Improvements

1. **Strict Input Contract**:
   - 6 linear-power SAR channels (pre-event secondary 1, pre-event secondary 2, and post-event flood scene).
   - DEM is preserved for downstream hydrological routing and is not fed into the SAR optical encoder.
2. **Fixed Metric Checkpoint Selection**:
   - Checkpoints are saved using **fixed argmax flood IoU** on validation data to prevent optimistic threshold overfitting.
3. **Post-Hoc Threshold Calibration**:
   - Optimal probability threshold ($\tau = 0.30$) is calibrated once on the validation set using the selected best checkpoint.
4. **Resumable Training & Determinism**:
   - Atomic checkpoints with step-level state, optimizer replay, and deterministic seed (`seed=42`).
5. **Data Scaling & Normalization**:
   - Clipped to maximum linear power of `0.15`.
   - Mean: `[0.0953, 0.0264, 0.0953, 0.0264, 0.0953, 0.0264]`
   - Std: `[0.0427, 0.0215, 0.0427, 0.0215, 0.0427, 0.0215]`

---

## 🚀 Getting Started

### 1. Prerequisites
Install Git LFS to pull model weights:
```bash
git lfs install
git clone https://github.com/Vd-adasul/iitm-hackathon.git
cd iitm-hackathon
git lfs pull
```

### 2. Python Environment
```bash
pip install segmentation-models-pytorch scipy scikit-learn matplotlib numpy pandas torch torchvision
```

### 3. Loading the Model for Inference
```python
import torch
import segmentation_models_pytorch as smp

# Initialize U-Net architecture
model = smp.Unet(
    encoder_name="resnet50",
    encoder_weights=None,
    in_channels=6,
    classes=3,
)

# Load calibrated weights
weights_path = "outputs_e2e/best_unet_resnet50.pth"
state_dict = torch.load(weights_path, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# Threshold for flood class (class index 2)
FLOOD_THRESHOLD = 0.30
```
