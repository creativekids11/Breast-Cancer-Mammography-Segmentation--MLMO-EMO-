# MLMO-EMO Quick Reference Guide

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install torch torchvision opencv-python albumentations segmentation-models-pytorch pandas numpy tqdm tensorboard matplotlib scikit-learn
```

### 2. Train Model (Easiest Way)
**Windows:**
```bash
train_mlmo_emo.bat
```

**Linux/Mac:**
```bash
./train_mlmo_emo.sh
```

### 3. Run Inference
```bash
python inference_mlmo_emo.py --weights checkpoints_mlmo_emo/best_model.pth --image-dir test_images --output-dir results
```

---

## 📝 Common Commands

### Training Commands

#### Train with default hyperparameters
```powershell
python train_mlmo_emo.py `
  --data-dir segmentation_data/train_valid `
  --csv-path unified_segmentation_dataset.csv `
  --use-csv
```

#### Train with custom settings
```powershell
python train_mlmo_emo.py `
  --data-dir segmentation_data/train_valid `
  --csv-path unified_segmentation_dataset.csv `
  --use-csv `
  --epochs 150 `
  --batch-size 12 `
  --lr 3e-4 `
  --img-size 512 `
  --num-particles 3 `
  --emo-iterations 3 `
  --encoder resnet34
```

#### Resume training
```powershell
python train_mlmo_emo.py `
  --data-dir segmentation_data/train_valid `
  --resume checkpoints_mlmo_emo/checkpoint_epoch_50.pth
```

### Inference Commands

#### Single image
```powershell
python inference_mlmo_emo.py `
  --weights checkpoints_mlmo_emo/best_model.pth `
  --image path/to/image.png `
  --output-dir predictions
```

#### Batch inference
```powershell
python inference_mlmo_emo.py `
  --weights checkpoints_mlmo_emo/best_model.pth `
  --image-dir path/to/images `
  --output-dir predictions `
  --visualize
```

#### Evaluate on test set
```powershell
python inference_mlmo_emo.py `
  --weights checkpoints_mlmo_emo/best_model.pth `
  --image-dir test/images `
  --mask-dir test/masks `
  --evaluate
```

---

## ⚙️ Hyperparameter Reference

### Model Architecture
| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `--encoder` | resnet34 | resnet18, resnet34, resnet50, efficientnet-b0 | Backbone encoder |
| `--num-particles` | 3 | 2-5 | Number of EM particles |
| `--emo-iterations` | 3 | 2-5 | EMO refinement iterations |

### Training
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `--epochs` | 150 | 50-300 | Training epochs |
| `--batch-size` | 12 | 4-32 | Batch size |
| `--lr` | 3e-4 | 1e-5 to 1e-3 | Learning rate |
| `--img-size` | 512 | 256-1024 | Input image size |
| `--num-workers` | 4 | 0-16 | Data loading threads |

### Loss Weights
| Parameter | Default | Range | Purpose |
|-----------|---------|-------|---------|
| `--dice-weight` | 1.0 | 0.5-2.0 | Segmentation overlap |
| `--bce-weight` | 1.0 | 0.5-2.0 | Pixel accuracy |
| `--boundary-weight` | 0.5 | 0.1-1.0 | Edge quality |
| `--homogeneity-weight` | 0.3 | 0.1-0.5 | Region consistency |

---

## 🎯 Tuning Guide

### For Better Accuracy
```powershell
--num-particles 4 `
--emo-iterations 4 `
--encoder resnet50 `
--boundary-weight 0.7
```

### For Faster Training
```powershell
--num-particles 2 `
--emo-iterations 2 `
--encoder resnet18 `
--batch-size 16
```

### For Better Boundaries
```powershell
--boundary-weight 0.8 `
--emo-iterations 4
```

### For Small Dataset
```powershell
--epochs 200 `
--lr 1e-4 `
--encoder resnet18
```

---

## 📊 Monitoring

### View training progress
```bash
tensorboard --logdir runs/mlmo_emo_segmentation
```

### Check checkpoint metrics
```python
import torch
checkpoint = torch.load('checkpoints_mlmo_emo/best_model.pth')
print(checkpoint['metrics'])
```

---

## 🐛 Troubleshooting

### Out of Memory
```powershell
--batch-size 4 `
--img-size 384 `
--num-particles 2
```

### Poor Results
```powershell
--epochs 200 `
--num-particles 4 `
--emo-iterations 4 `
--lr 1e-4
```

### Slow Training
```powershell
--num-workers 8 `
--batch-size 16
```

---

## 📁 Output Structure

```
checkpoints_mlmo_emo/
├── best_model.pth                    # Best model weights
├── checkpoint_epoch_N.pth            # Regular checkpoints
└── visualizations_epoch_N/           # Sample predictions
    ├── sample_0.png
    └── ...

predictions/
├── image_name_mask.png               # Binary mask
└── image_name_visualization.png      # Overlay visualization

runs/mlmo_emo_segmentation/           # TensorBoard logs
```

---

## 🔬 Model Testing

### Quick test on dummy data
```bash
python mlmo_emo_segmentation.py
```

### Validate model architecture
```python
from mlmo_emo_segmentation import create_mlmo_emo_model
model = create_mlmo_emo_model()
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

---

## 📈 Expected Performance

| Metric | Expected Range |
|--------|----------------|
| **Dice Coefficient** | 0.85 - 0.92 |
| **IoU** | 0.75 - 0.85 |
| **Pixel Accuracy** | 0.90 - 0.96 |

*Performance varies based on dataset quality and hyperparameters*

---

## 💡 Tips

1. **Start with default settings** - they work well for most cases
2. **Monitor TensorBoard** - watch for overfitting
3. **Use data augmentation** - built into training pipeline
4. **Save checkpoints frequently** - training can be interrupted
5. **Evaluate on validation set** - don't overtune on training data
6. **Adjust threshold** - 0.5 may not be optimal for your data

---

## 📞 Need Help?

1. Check `MLMO_EMO_DOCUMENTATION.md` for detailed info
2. View training logs in TensorBoard
3. Test on small dataset first
4. Ensure GPU is detected: `python -c "import torch; print(torch.cuda.is_available())"`
