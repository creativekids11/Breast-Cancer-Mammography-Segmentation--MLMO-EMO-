# Multi-Level Multi-Objective Electromagnetism-like Optimization (MLMO-EMO)
## Breast Cancer Mammography Segmentation

## Overview

This implementation introduces a novel **Multi-Level Multi-Objective Electromagnetism-like Optimization (MLMO-EMO)** approach for end-to-end breast cancer segmentation from mammography images. Unlike traditional cascade approaches, MLMO-EMO processes the entire image directly using electromagnetic field principles combined with deep learning.

### Key Features

✨ **Multi-Level Feature Extraction**: Captures image information at multiple scales simultaneously

🧲 **Electromagnetism-like Optimization**: Uses attraction-repulsion forces between particles to refine segmentation

🎯 **Multi-Objective Optimization**: Simultaneously optimizes for:
- Segmentation accuracy
- Boundary smoothness
- Region homogeneity

🔄 **End-to-End Processing**: Direct whole-image segmentation without cascading stages

## Architecture

### 1. Multi-Level Feature Extractor
Extracts hierarchical features using a pretrained encoder (ResNet34 by default):
```
Input Image [512x512x3]
    ↓
ResNet34 Encoder
    ↓
Multi-Scale Features [Level 1-5]
```

### 2. Electromagnetic Particles
Each particle represents a potential segmentation solution:
- **Position**: Encoded feature representation
- **Charge**: Segmentation confidence
- **Forces**: Attraction for similar regions, repulsion at boundaries

```python
# Particle interaction
Force(i,j) = f(Position_i, Position_j) * (Charge_i + Charge_j) / 2
```

### 3. EMO Refinement Process
Iterative optimization through particle interactions:
```
for iteration in range(emo_iterations):
    for each particle i:
        total_force = sum(forces from other particles)
        position_i = position_i + learning_rate * total_force
```

### 4. Multi-Objective Heads
Three specialized heads optimize different aspects:
- **Accuracy Head**: Primary segmentation quality
- **Boundary Head**: Edge smoothness and definition
- **Homogeneity Head**: Region consistency

Final output is a weighted combination with learnable weights.

## Installation

### Requirements
```bash
pip install torch torchvision opencv-python albumentations segmentation-models-pytorch pandas numpy tqdm tensorboard matplotlib scikit-learn
```

### Quick Start
The easiest way to get started is using the provided batch files:

**Windows:**
```bash
train_mlmo_emo.bat
```

**Linux/Mac:**
```bash
chmod +x train_mlmo_emo.sh
./train_mlmo_emo.sh
```

## Usage

### Training

#### Basic Training
```powershell
python train_mlmo_emo.py `
  --data-dir segmentation_data/train_valid `
  --csv-path unified_segmentation_dataset.csv `
  --use-csv `
  --epochs 150 `
  --batch-size 12 `
  --lr 3e-4
```

#### Advanced Training Options
```powershell
python train_mlmo_emo.py `
  --data-dir segmentation_data/train_valid `
  --csv-path unified_segmentation_dataset.csv `
  --use-csv `
  --encoder resnet34 `
  --num-particles 3 `
  --emo-iterations 3 `
  --epochs 150 `
  --batch-size 12 `
  --lr 3e-4 `
  --img-size 512 `
  --num-workers 4 `
  --dice-weight 1.0 `
  --bce-weight 1.0 `
  --boundary-weight 0.5 `
  --homogeneity-weight 0.3 `
  --checkpoint-dir checkpoints_mlmo_emo `
  --logdir runs/mlmo_emo_segmentation
```

#### Training Parameters

**Model Architecture:**
- `--encoder`: Backbone encoder (resnet18, resnet34, resnet50, efficientnet-b0, etc.)
- `--num-particles`: Number of electromagnetic particles (default: 3)
- `--emo-iterations`: EMO refinement iterations (default: 3)

**Training Hyperparameters:**
- `--epochs`: Training epochs (default: 150)
- `--batch-size`: Batch size (default: 12)
- `--lr`: Learning rate (default: 3e-4)
- `--img-size`: Input image size (default: 512)
- `--num-workers`: Data loading workers (default: 4)

**Loss Weights:**
- `--dice-weight`: Weight for Dice loss (default: 1.0)
- `--bce-weight`: Weight for BCE loss (default: 1.0)
- `--boundary-weight`: Weight for boundary loss (default: 0.5)
- `--homogeneity-weight`: Weight for homogeneity loss (default: 0.3)

**Data:**
- `--data-dir`: Directory with training data
- `--csv-path`: CSV file with image/mask pairs
- `--use-csv`: Use CSV instead of directory structure

### Inference

#### Single Image
```powershell
python inference_mlmo_emo.py `
  --weights checkpoints_mlmo_emo/best_model.pth `
  --image path/to/image.png `
  --output-dir predictions `
  --visualize
```

#### Batch Processing
```powershell
python inference_mlmo_emo.py `
  --weights checkpoints_mlmo_emo/best_model.pth `
  --image-dir path/to/images `
  --output-dir predictions `
  --visualize
```

#### Evaluation with Ground Truth
```powershell
python inference_mlmo_emo.py `
  --weights checkpoints_mlmo_emo/best_model.pth `
  --image-dir path/to/images `
  --mask-dir path/to/masks `
  --output-dir predictions `
  --evaluate
```

#### Inference Parameters
- `--weights`: Path to trained model weights (required)
- `--encoder`: Encoder used during training (default: resnet34)
- `--num-particles`: Number of particles used during training (default: 3)
- `--emo-iterations`: EMO iterations used during training (default: 3)
- `--img-size`: Input image size (default: 512)
- `--threshold`: Binary segmentation threshold (default: 0.5)
- `--visualize`: Create visualization images

## Dataset Format

### Option 1: CSV Format
Create a CSV file with image and mask paths:
```csv
image_path,mask_path
images/img001.png,masks/mask001.png
images/img002.png,masks/mask002.png
```

Use with `--use-csv` flag.

### Option 2: Directory Structure
```
segmentation_data/
├── train/
│   ├── images/
│   │   ├── img001.png
│   │   └── img002.png
│   └── masks/
│       ├── mask001.png
│       └── mask002.png
└── val/
    ├── images/
    └── masks/
```

Or simpler structure:
```
segmentation_data/
├── images/
│   ├── img001.png
│   └── img002.png
└── masks/
    ├── mask001.png
    └── mask002.png
```

## Model Architecture Details

### Electromagnetic Particle Module
```python
class ElectromagneticParticle(nn.Module):
    - Position Encoder: Conv layers for feature encoding
    - Charge Network: Calculates segmentation confidence
    - Force Network: Computes particle interactions
```

### Multi-Objective Optimization
```python
class MultiObjectiveOptimizationModule(nn.Module):
    - Accuracy Head: Primary segmentation
    - Boundary Head: Edge quality
    - Homogeneity Head: Region consistency
    - Learnable Weights: Adaptive objective combination
```

### Complete Model Flow
```
Input [B, 3, H, W]
    ↓
Multi-Level Feature Extraction
    ↓
Decoder (UNet-style)
    ↓
Electromagnetic Particles (3x)
    ↓
EMO Refinement (3 iterations)
    ↓
Feature Fusion
    ↓
Multi-Objective Optimization
    ↓
Final Segmentation [B, 1, H, W]
```

## Loss Function

The MLMO-EMO loss combines multiple objectives:

```
Total Loss = 
    dice_weight * DiceLoss +
    bce_weight * BCELoss +
    boundary_weight * BoundaryLoss +
    homogeneity_weight * HomogeneityLoss
```

**Components:**
1. **Dice Loss**: Overlap between prediction and ground truth
2. **BCE Loss**: Pixel-wise binary cross-entropy
3. **Boundary Loss**: Gradient consistency at edges
4. **Homogeneity Loss**: Variance within predicted regions

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir runs/mlmo_emo_segmentation
```

Tracked metrics:
- Training/Validation Loss
- Dice Coefficient
- IoU (Intersection over Union)
- Learning Rate
- Individual loss components

### Checkpoints
- Regular checkpoints: `checkpoint_epoch_N.pth`
- Best model: `best_model.pth` (highest validation Dice)
- Visualizations: `visualizations_epoch_N/` (every 10 epochs)

## Performance Optimization

### GPU Acceleration
- Automatically uses CUDA if available
- Mixed precision training can be added for faster training
- Adjust `--batch-size` based on GPU memory

### CPU Training
- Set `--num-workers 0` if experiencing issues
- Reduce `--batch-size` if memory limited
- Expected to be 2-3x slower than GPU

### Hyperparameter Tuning

**For Better Accuracy:**
- Increase `--num-particles` (4-5)
- Increase `--emo-iterations` (4-5)
- Use larger encoder (resnet50, efficientnet-b4)

**For Faster Training:**
- Decrease `--num-particles` (2)
- Decrease `--emo-iterations` (2)
- Use smaller encoder (resnet18)

**For Better Boundaries:**
- Increase `--boundary-weight` (0.7-1.0)
- Increase `--emo-iterations`

## Advantages over Cascade Approach

| Aspect | Cascade Model | MLMO-EMO |
|--------|---------------|----------|
| **Training** | Two-stage sequential | Single end-to-end |
| **Inference** | Multiple passes | Single forward pass |
| **Error Propagation** | Stage 1 errors affect Stage 2 | Direct optimization |
| **Optimization** | Single objective per stage | Multi-objective |
| **Computational Cost** | Higher (2 models) | Lower (1 model) |
| **Flexibility** | Rigid pipeline | Adaptive optimization |

## Troubleshooting

### Out of Memory (OOM)
- Reduce `--batch-size`
- Reduce `--img-size`
- Reduce `--num-particles`

### Poor Segmentation Quality
- Increase `--epochs`
- Adjust loss weights
- Increase `--num-particles` and `--emo-iterations`
- Use larger encoder

### Slow Training
- Increase `--num-workers` (up to CPU cores)
- Reduce `--img-size` (but may hurt accuracy)
- Ensure GPU is being used

## Citation

If you use this implementation, please cite:

```bibtex
@software{mlmo_emo_segmentation,
  title={Multi-Level Multi-Objective Electromagnetism-like Optimization for Breast Cancer Segmentation},
  year={2025},
  note={Implementation of MLMO-EMO for mammography segmentation}
}
```

## License

See `LICENSE.md` for details.

## Acknowledgments

- Segmentation Models PyTorch for encoder implementations
- Albumentations for data augmentation
- PyTorch team for the deep learning framework

## Contact & Support

For questions, issues, or contributions, please open an issue in the repository.

---

**Note**: This is an advanced implementation combining electromagnetic optimization principles with deep learning. For production use, ensure thorough validation on your specific dataset.
