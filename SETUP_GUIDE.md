# Setup Guide

This guide will walk you through setting up the MLMO-EMO Segmentation project from scratch.

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8 or higher**
  - Check: `python --version`
  - Download: https://www.python.org/downloads/

- **CUDA Toolkit** (for GPU acceleration)
  - Check: `nvidia-smi`
  - Download: https://developer.nvidia.com/cuda-downloads
  - Recommended: CUDA 11.8 or 12.1

- **Git**
  - Check: `git --version`
  - Download: https://git-scm.com/downloads

- **Minimum System Requirements**
  - RAM: 16GB minimum (32GB recommended)
  - GPU: NVIDIA GPU with 8GB+ VRAM (RTX 3060 or better)
  - Storage: 20GB free space (for code + data + checkpoints)

## 🚀 Installation Methods

### Method 1: Quick Install (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/mlmo-emo-segmentation.git
cd mlmo-emo-segmentation

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Method 2: Conda Install

```bash
# Clone the repository
git clone https://github.com/yourusername/mlmo-emo-segmentation.git
cd mlmo-emo-segmentation

# Create conda environment
conda create -n mlmo-emo python=3.8
conda activate mlmo-emo

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install segmentation-models-pytorch albumentations opencv-python pillow tqdm tensorboard
```

### Method 3: Development Install

```bash
# Clone your fork
git clone https://github.com/your-username/mlmo-emo-segmentation.git
cd mlmo-emo-segmentation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode
pip install -e .

# Install development dependencies
pip install pytest black flake8 mypy
```

## ✅ Verify Installation

### 1. Check Python and Dependencies

```bash
python --version  # Should be 3.8+
pip list | grep torch  # Should show torch, torchvision, torchaudio
pip list | grep segmentation  # Should show segmentation-models-pytorch
```

### 2. Test CUDA Availability

```python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Expected output:
```
CUDA available: True
CUDA version: 11.8
GPU: NVIDIA GeForce RTX 3090
```

### 3. Run Quick Test

```bash
python examples/quick_test.py
```

Expected output:
```
Testing MLMO-EMO Model...
Model created successfully!
Model Parameters: 27,366,633
Testing forward pass...
Input shape: torch.Size([2, 1, 256, 256])
Output shape: torch.Size([2, 1, 256, 256])
✓ Forward pass successful!
Testing loss calculation...
Loss: 0.6234
✓ Loss calculation successful!
All tests passed! ✓
```

## 📁 Dataset Setup

### Option 1: CBIS-DDSM Dataset

1. **Download CBIS-DDSM**
   - Visit: https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM
   - Download the dataset (requires TCIA downloader)

2. **Organize Directory Structure**
   ```
   data/
   ├── train/
   │   ├── images/
   │   │   ├── mass_001.png
   │   │   └── ...
   │   └── masks/
   │       ├── mass_001.png
   │       └── ...
   └── val/
       ├── images/
       └── masks/
   ```

3. **Preprocess Images** (if needed)
   ```bash
   python dataset_process.py --input raw_data/ --output data/
   ```

### Option 2: Mini-DDSM Dataset

1. **Download Mini-DDSM**
   ```bash
   # Download from provided link
   wget [mini-ddsm-link]
   unzip mini-ddsm.zip -d data/
   ```

2. **Verify Structure**
   ```bash
   ls data/train/images/  # Should show image files
   ls data/train/masks/   # Should show mask files
   ```

### Option 3: Custom Dataset

1. **Prepare Your Images**
   - Format: PNG, JPG, or TIFF
   - Type: Grayscale mammography images
   - Size: Any size (will be resized during training)

2. **Prepare Masks**
   - Format: PNG (binary: 0 for background, 255 for lesion)
   - Same filenames as corresponding images
   - Same dimensions as images

3. **Organize Files**
   ```
   data/
   ├── train/
   │   ├── images/
   │   └── masks/
   └── val/
       ├── images/
       └── masks/
   ```

## 🎯 Quick Start

### 1. Test Preprocessing

```bash
python examples/test_preprocessing.py
```

This will generate visualization images showing:
- Original image
- Linear normalization
- Sigmoid normalization
- Median filtering
- CLAHE enhancement

### 2. Run Training (Quick Test)

```bash
# Windows
scripts\train_mlmo_emo.bat

# Linux/Mac
bash scripts/train_mlmo_emo.sh
```

Or with custom parameters:

```bash
python train_mlmo_emo.py \
    --train-images data/train/images \
    --train-masks data/train/masks \
    --val-images data/val/images \
    --val-masks data/val/masks \
    --epochs 5 \
    --batch-size 2
```

### 3. Monitor Training

Open a new terminal and run:

```bash
tensorboard --logdir=runs
```

Then open your browser to: `http://localhost:6006`

### 4. Run Inference

```bash
python inference_mlmo_emo.py \
    --weights checkpoints/best_model.pth \
    --images test_images/ \
    --output results/
```

## 🔧 Troubleshooting

### Issue: CUDA Out of Memory

**Solution:**
```bash
# Reduce batch size
python train_mlmo_emo.py --batch-size 1

# Reduce image size
python train_mlmo_emo.py --image-size 256

# Reduce number of particles
python train_mlmo_emo.py --num-particles 2
```

### Issue: Module Not Found

**Solution:**
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Or install specific package
pip install segmentation-models-pytorch
```

### Issue: CUDA Not Available

**Solution:**
1. Check NVIDIA driver: `nvidia-smi`
2. Reinstall PyTorch with CUDA:
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### Issue: Training Loss is NaN

**Solution:**
1. Check data normalization
2. Reduce learning rate: `--lr 0.00001`
3. Verify masks are binary (0 and 1)

### Issue: Import Error for cv2

**Solution:**
```bash
pip uninstall opencv-python
pip install opencv-python-headless
```

## 📚 Next Steps

After successful setup:

1. **Read Documentation**
   - [MLMO_EMO_DOCUMENTATION.md](docs/MLMO_EMO_DOCUMENTATION.md)
   - [PREPROCESSING_DOCUMENTATION.md](docs/PREPROCESSING_DOCUMENTATION.md)
   - [QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)

2. **Experiment with Parameters**
   - Try different encoders
   - Test different preprocessing methods
   - Adjust augmentation strength

3. **Train on Your Data**
   - Prepare your dataset
   - Start with small epochs for testing
   - Monitor metrics on TensorBoard

4. **Evaluate Results**
   - Run inference on test set
   - Calculate comprehensive metrics
   - Visualize predictions

## 🆘 Getting Help

If you encounter issues:

1. **Check Documentation**: Most common issues are covered in docs
2. **Search Issues**: GitHub issues may have solutions
3. **Open Issue**: Describe your problem with:
   - System information
   - Error message
   - Steps to reproduce
4. **Contact**: [your-email@example.com]

## 🎓 Learning Resources

- **PyTorch Tutorial**: https://pytorch.org/tutorials/
- **Medical Image Segmentation**: https://arxiv.org/abs/1505.04597
- **CBIS-DDSM Paper**: https://www.nature.com/articles/sdata2017177
- **Albumentations Docs**: https://albumentations.ai/docs/

---

**Happy Segmenting! 🎗️**
