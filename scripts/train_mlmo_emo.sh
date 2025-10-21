#!/bin/bash

# Training Script for MLMO-EMO Breast Cancer Segmentation Model
# Multi-Level Multi-Objective Electromagnetism-like Optimization

echo "========================================"
echo "MLMO-EMO Segmentation Model Training"
echo "========================================"
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python not found! Please install Python first."
    exit 1
fi

echo "[1/3] Checking required packages..."
python -c "import torch, cv2, albumentations, segmentation_models_pytorch, pandas" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required packages..."
    pip install torch torchvision opencv-python albumentations segmentation-models-pytorch pandas numpy tqdm tensorboard matplotlib scikit-learn
else
    echo "All packages are installed!"
fi

echo ""
echo "[2/3] Training configuration:"
echo "  - Model: Multi-Level Multi-Objective EMO"
echo "  - Epochs: 150"
echo "  - Electromagnetic Particles: 3"
echo "  - EMO Iterations: 3"
echo "  - Image Size: 512x512"
echo "  - Batch Size: 12"
echo "  - Learning Rate: 3e-4"
echo "  - Estimated training time: ~10-15 hours (GPU) or 30-48 hours (CPU)"
echo ""

# Check if CUDA is available
python -c "import torch; print('GPU Available!' if torch.cuda.is_available() else 'Using CPU (slower)')"
echo ""

echo "[3/3] Starting MLMO-EMO training..."
echo ""

# Run training with MLMO-EMO model
python train_mlmo_emo.py \
    --data-dir segmentation_data/train_valid \
    --csv-path unified_segmentation_dataset.csv \
    --use-csv \
    --encoder resnet34 \
    --num-particles 3 \
    --emo-iterations 3 \
    --epochs 150 \
    --batch-size 12 \
    --lr 3e-4 \
    --img-size 512 \
    --num-workers 4 \
    --dice-weight 1.0 \
    --bce-weight 1.0 \
    --boundary-weight 0.5 \
    --homogeneity-weight 0.3 \
    --checkpoint-dir checkpoints_mlmo_emo \
    --logdir runs/mlmo_emo_segmentation

echo ""
echo "========================================"
echo "Training Complete!"
echo "========================================"
echo ""
echo "Checkpoints saved to: checkpoints_mlmo_emo/"
echo ""
echo "To view training progress:"
echo "  tensorboard --logdir runs/mlmo_emo_segmentation"
echo ""
echo "To run inference:"
echo "  python inference_mlmo_emo.py --weights checkpoints_mlmo_emo/best_model.pth --image-dir path/to/images --output-dir predictions"
echo ""
