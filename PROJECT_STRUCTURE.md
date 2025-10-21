# Project Structure

This document provides an overview of the MLMO-EMO Segmentation project structure.

## 📁 Directory Layout

```
mlmo-emo-release/
├── .github/                    # GitHub specific files
│   └── workflows/
│       └── ci.yml             # CI/CD pipeline configuration
│
├── assets/                     # Visual assets and images
│   ├── median_filtering_comparison.png
│   ├── normalization_comparison.png
│   ├── preprocessing_pipeline.png
│   └── preprocessing_variants.png
│
├── docs/                       # Documentation files
│   ├── MLMO_EMO_DOCUMENTATION.md      # Complete technical documentation
│   ├── PREPROCESSING_DOCUMENTATION.md  # Preprocessing methods and equations
│   └── QUICK_REFERENCE.md              # Quick command reference
│
├── examples/                   # Example and test scripts
│   ├── quick_test.py          # Quick model test
│   └── test_preprocessing.py  # Preprocessing visualization
│
├── scripts/                    # Launcher scripts
│   ├── train_mlmo_emo.bat     # Windows training script
│   └── train_mlmo_emo.sh      # Linux/Mac training script
│
├── mlmo_emo_segmentation.py   # Core model implementation
├── train_mlmo_emo.py          # Training pipeline
├── inference_mlmo_emo.py      # Inference pipeline
├── dataset_process.py         # Dataset processing and loading
│
├── .gitignore                 # Git ignore rules
├── CHANGELOG.md               # Version history and changes
├── CODE_OF_CONDUCT.md         # Community guidelines
├── CONTRIBUTING.md            # Contribution guidelines
├── LICENSE.md                 # MIT License
├── README.md                  # Main project documentation
├── requirements.txt           # Python dependencies
└── SETUP_GUIDE.md            # Detailed setup instructions
```

## 📄 Core Files

### Model Implementation

**`mlmo_emo_segmentation.py`** (442 lines)
- `ElectromagneticParticle`: Individual particle for feature extraction
- `MultiLevelFeatureExtractor`: Hierarchical feature extraction
- `MultiObjectiveOptimizationModule`: EMO-based refinement
- `MLMOEMOSegmentation`: Main model class
- `MLMOEMOLoss`: Multi-objective loss function

**Key Components:**
```python
MLMOEMOSegmentation(
    encoder_name='resnet34',      # Backbone encoder
    encoder_weights='imagenet',    # Pre-trained weights
    in_channels=1,                 # Grayscale input
    num_particles=3,               # Number of particles
    emo_iterations=3               # Optimization iterations
)
```

### Training Pipeline

**`train_mlmo_emo.py`** (624 lines)
- Complete training loop with validation
- Data augmentation (light/medium/strong)
- Early stopping with patience
- Model checkpointing (best + last)
- TensorBoard logging
- Comprehensive metrics calculation

**Key Functions:**
- `calculate_metrics()`: Computes 6 segmentation metrics
- `train_epoch()`: Single training epoch
- `validate_epoch()`: Validation with metrics
- `create_data_loaders()`: Dataset loading
- `main()`: Entry point with argument parsing

### Inference Pipeline

**`inference_mlmo_emo.py`** (369 lines)
- Batch inference on directories
- Visualization with overlays
- Evaluation with ground truth
- Comprehensive metrics reporting

**Key Functions:**
- `inference_on_directory()`: Batch prediction
- `evaluate_with_ground_truth()`: Calculate metrics
- `visualize_prediction()`: Create overlay images

### Dataset Processing

**`dataset_process.py`** (300+ lines)
- MammographyDataset class
- Preprocessing methods:
  - Linear normalization (Equation 1)
  - Sigmoid normalization (Equation 2)
  - Median filtering (Equation 3)
  - CLAHE with adaptive clip limit (Equations 4-8)
- Data augmentation pipelines

## 📚 Documentation Files

### Main Documentation

**`README.md`** (~350 lines)
- Project overview and features
- Installation instructions
- Quick start guide
- Training and inference examples
- API reference
- Performance benchmarks

### Technical Documentation

**`docs/MLMO_EMO_DOCUMENTATION.md`** (650+ lines)
- Detailed architecture explanation
- Mathematical formulations
- Component descriptions
- Training strategies
- Troubleshooting guide

**`docs/PREPROCESSING_DOCUMENTATION.md`** (400+ lines)
- All preprocessing equations (1-8)
- Implementation details
- Usage examples
- Visual comparisons

**`docs/QUICK_REFERENCE.md`** (200+ lines)
- Common commands
- Parameter reference
- Quick troubleshooting

### Setup and Contributing

**`SETUP_GUIDE.md`** (300+ lines)
- Step-by-step setup instructions
- System requirements
- Dataset preparation
- Troubleshooting common issues
- Next steps after setup

**`CONTRIBUTING.md`** (400+ lines)
- Contribution guidelines
- Code style guide
- Pull request process
- Development setup
- Testing guidelines

### Project Management

**`CHANGELOG.md`**
- Version history
- Feature additions
- Bug fixes
- Future roadmap

**`CODE_OF_CONDUCT.md`**
- Community standards
- Behavior expectations
- Enforcement guidelines

## 🔧 Configuration Files

### Python Dependencies

**`requirements.txt`**
```
torch>=2.0.0
segmentation-models-pytorch>=0.3.3
albumentations>=1.3.1
opencv-python>=4.8.0
tensorboard>=2.14.0
# ... and more
```

### Git Configuration

**`.gitignore`**
- Python artifacts (__pycache__, *.pyc)
- Model weights (*.pth, *.pt)
- Data directories
- TensorBoard logs
- Virtual environments
- IDE files

### CI/CD

**`.github/workflows/ci.yml`**
- Automated testing on push/PR
- Multi-Python version testing (3.8, 3.9, 3.10)
- Code style checking
- Documentation validation

## 🚀 Example Scripts

### Quick Test

**`examples/quick_test.py`** (~50 lines)
Tests:
- Model creation
- Forward pass
- Loss calculation
- Output shapes

### Preprocessing Test

**`examples/test_preprocessing.py`** (~100 lines)
Generates visualizations for:
- Original image
- Linear normalization
- Sigmoid normalization
- Median filtering
- CLAHE enhancement

## 📦 Launcher Scripts

### Windows

**`scripts/train_mlmo_emo.bat`**
```batch
python train_mlmo_emo.py ^
    --train-images data/train/images ^
    --train-masks data/train/masks ^
    ...
```

### Linux/Mac

**`scripts/train_mlmo_emo.sh`**
```bash
python train_mlmo_emo.py \
    --train-images data/train/images \
    --train-masks data/train/masks \
    ...
```

## 🎨 Assets

### Preprocessing Visualizations

- **preprocessing_pipeline.png**: Complete pipeline overview
- **normalization_comparison.png**: Linear vs Sigmoid
- **median_filtering_comparison.png**: Before/after filtering
- **preprocessing_variants.png**: All methods side-by-side

## 📊 Generated Directories

These directories are created during usage (not in repo):

```
mlmo-emo-release/
├── data/                      # Dataset (user provided)
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   └── val/
│       ├── images/
│       └── masks/
│
├── checkpoints/               # Saved models
│   ├── best_model.pth
│   └── last_checkpoint.pth
│
├── runs/                      # TensorBoard logs
│   └── experiment_name/
│
└── results/                   # Inference outputs
    ├── predictions/
    └── visualizations/
```

## 🔍 File Size Reference

| File | Lines | Purpose |
|------|-------|---------|
| mlmo_emo_segmentation.py | 442 | Model implementation |
| train_mlmo_emo.py | 624 | Training pipeline |
| inference_mlmo_emo.py | 369 | Inference pipeline |
| dataset_process.py | 300+ | Dataset processing |
| README.md | 350+ | Main documentation |
| MLMO_EMO_DOCUMENTATION.md | 650+ | Technical docs |
| SETUP_GUIDE.md | 300+ | Setup instructions |
| CONTRIBUTING.md | 400+ | Contribution guide |

## 🎯 Entry Points

### For Users

1. **Setup**: `SETUP_GUIDE.md`
2. **Quick Start**: `README.md` → Quick Start section
3. **Training**: `python train_mlmo_emo.py` or `scripts/train_mlmo_emo.bat`
4. **Inference**: `python inference_mlmo_emo.py`

### For Developers

1. **Architecture**: `docs/MLMO_EMO_DOCUMENTATION.md`
2. **Contributing**: `CONTRIBUTING.md`
3. **Testing**: `examples/quick_test.py`
4. **Code**: `mlmo_emo_segmentation.py`

### For Researchers

1. **Model Details**: `docs/MLMO_EMO_DOCUMENTATION.md`
2. **Preprocessing**: `docs/PREPROCESSING_DOCUMENTATION.md`
3. **Metrics**: `README.md` → Metrics section
4. **Results**: `README.md` → Performance Benchmarks

## 📝 Notes

- All Python files include docstrings
- All functions have type hints
- Code follows PEP 8 style (with 100 char line limit)
- Documentation uses Markdown with proper formatting
- Examples are tested and functional

## 🔗 Quick Links

- **Main README**: Start here for overview
- **Setup Guide**: For installation
- **API Docs**: `docs/MLMO_EMO_DOCUMENTATION.md`
- **Contributing**: `CONTRIBUTING.md`
- **License**: `LICENSE.md`

---

**Last Updated**: 2025-10-21
**Version**: 1.0.0
