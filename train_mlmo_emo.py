"""
Training script for Multi-Level Multi-Objective Electromagnetism-like Optimization (MLMO-EMO)
Breast Cancer Mammography Segmentation Model
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from mlmo_emo_segmentation import create_mlmo_emo_model, MLMOEMOLoss


class MammographyDataset(Dataset):
    """Dataset for mammography images with segmentation masks."""
    
    def __init__(self, image_paths, mask_paths, transform=None, img_size=512):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.img_size = img_size
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = self.mask_paths[idx]
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not load mask: {mask_path}")
        
        # Ensure mask is binary
        mask = (mask > 127).astype(np.uint8)
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Ensure mask has correct shape [1, H, W]
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        
        mask = mask.float()
        
        return image, mask


def get_training_augmentation(img_size=512):
    """Get training augmentation pipeline."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_validation_augmentation(img_size=512):
    """Get validation augmentation pipeline."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def load_dataset_from_csv(csv_path, data_dir, test_size=0.2):
    """Load dataset from CSV file."""
    df = pd.read_csv(csv_path)
    
    image_paths = []
    mask_paths = []
    
    for _, row in df.iterrows():
        # Get image path
        if 'image_path' in row and pd.notna(row['image_path']):
            img_path = Path(data_dir) / row['image_path']
        elif 'image' in row and pd.notna(row['image']):
            img_path = Path(row['image'])
        elif 'image_file_path' in row and pd.notna(row['image_file_path']):
            img_path = Path(row['image_file_path'])
        else:
            img_path = None

        # Get mask path (support multiple possible column names)
        mask_path = None
        if 'mask_path' in row and pd.notna(row['mask_path']):
            mask_path = Path(data_dir) / row['mask_path']
        elif 'mask' in row and pd.notna(row['mask']):
            mask_path = Path(row['mask'])
        elif 'roi_mask_file_path' in row and pd.notna(row['roi_mask_file_path']):
            mask_path = Path(row['roi_mask_file_path'])

        if img_path is not None and mask_path is not None and img_path.exists() and mask_path.exists():
            image_paths.append(img_path)
            mask_paths.append(mask_path)
    
    print(f"Loaded {len(image_paths)} image-mask pairs from CSV")
    
    # Split into train and validation
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        image_paths, mask_paths, test_size=test_size, random_state=42
    )
    
    return train_imgs, val_imgs, train_masks, val_masks


def load_dataset_from_directory(data_dir):
    """Load dataset from directory structure."""
    data_path = Path(data_dir)
    
    # Look for train/val structure
    train_img_dir = data_path / "train" / "images"
    train_mask_dir = data_path / "train" / "masks"
    val_img_dir = data_path / "val" / "images"
    val_mask_dir = data_path / "val" / "masks"
    
    if train_img_dir.exists() and train_mask_dir.exists():
        train_imgs = sorted(list(train_img_dir.glob("*.png")) + list(train_img_dir.glob("*.jpg")))
        train_masks = sorted(list(train_mask_dir.glob("*.png")) + list(train_mask_dir.glob("*.jpg")))
        
        if val_img_dir.exists() and val_mask_dir.exists():
            val_imgs = sorted(list(val_img_dir.glob("*.png")) + list(val_img_dir.glob("*.jpg")))
            val_masks = sorted(list(val_mask_dir.glob("*.png")) + list(val_mask_dir.glob("*.jpg")))
        else:
            # Split training data
            train_imgs, val_imgs, train_masks, val_masks = train_test_split(
                train_imgs, train_masks, test_size=0.2, random_state=42
            )
    else:
        # Single directory structure
        img_dir = data_path / "images"
        mask_dir = data_path / "masks"
        
        if not img_dir.exists() or not mask_dir.exists():
            raise ValueError(f"Could not find images/masks in {data_dir}")
        
        all_imgs = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))
        all_masks = sorted(list(mask_dir.glob("*.png")) + list(mask_dir.glob("*.jpg")))
        
        train_imgs, val_imgs, train_masks, val_masks = train_test_split(
            all_imgs, all_masks, test_size=0.2, random_state=42
        )
    
    print(f"Loaded {len(train_imgs)} training and {len(val_imgs)} validation pairs")
    return train_imgs, val_imgs, train_masks, val_masks


def calculate_metrics(pred, target, threshold=0.5):
    """
    Calculate comprehensive segmentation metrics.
    
    Metrics:
    - Dice Coefficient (F1 Score)
    - Jaccard Coefficient (IoU)
    - Sensitivity (Recall/True Positive Rate)
    - Specificity (True Negative Rate)
    - Accuracy (Overall correctness)
    """
    pred = (torch.sigmoid(pred) > threshold).float()
    
    # Calculate confusion matrix components
    # TP: True Positive (predicted 1, actual 1)
    # TN: True Negative (predicted 0, actual 0)
    # FP: False Positive (predicted 1, actual 0)
    # FN: False Negative (predicted 0, actual 1)
    
    TP = ((pred == 1) & (target == 1)).float().sum()
    TN = ((pred == 0) & (target == 0)).float().sum()
    FP = ((pred == 1) & (target == 0)).float().sum()
    FN = ((pred == 0) & (target == 1)).float().sum()
    
    # Dice Coefficient (F1 Score)
    # Dice = 2 * TP / (2 * TP + FP + FN)
    dice = (2.0 * TP + 1e-7) / (2.0 * TP + FP + FN + 1e-7)
    
    # Jaccard Coefficient (IoU - Intersection over Union)
    # IoU = TP / (TP + FP + FN)
    jaccard = (TP + 1e-7) / (TP + FP + FN + 1e-7)
    
    # Sensitivity (Recall/True Positive Rate)
    # Sensitivity = TP / (TP + FN)
    sensitivity = (TP + 1e-7) / (TP + FN + 1e-7)
    
    # Specificity (True Negative Rate)
    # Specificity = TN / (TN + FP)
    specificity = (TN + 1e-7) / (TN + FP + 1e-7)
    
    # Accuracy (Overall correctness)
    # Accuracy = (TP + TN) / (TP + TN + FP + FN)
    accuracy = (TP + TN + 1e-7) / (TP + TN + FP + FN + 1e-7)
    
    return {
        'dice': dice.item(),
        'jaccard': jaccard.item(),
        'iou': jaccard.item(),  # IoU is same as Jaccard
        'sensitivity': sensitivity.item(),
        'specificity': specificity.item(),
        'accuracy': accuracy.item()
    }


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    
    running_loss = 0.0
    running_metrics = {
        'dice': 0.0, 
        'jaccard': 0.0, 
        'iou': 0.0,
        'sensitivity': 0.0, 
        'specificity': 0.0, 
        'accuracy': 0.0
    }
    
    pbar = tqdm(dataloader, desc=f"Training Epoch {epoch}")
    
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Calculate loss
        losses = criterion(outputs, masks)
        loss = losses['total']
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        metrics = calculate_metrics(outputs['segmentation'], masks)
        
        # Update running stats
        running_loss += loss.item()
        for key in running_metrics:
            running_metrics[key] += metrics[key]
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'dice': f"{metrics['dice']:.4f}",
            'sens': f"{metrics['sensitivity']:.4f}"
        })
    
    # Calculate epoch averages
    num_batches = len(dataloader)
    epoch_loss = running_loss / num_batches
    epoch_metrics = {k: v / num_batches for k, v in running_metrics.items()}
    
    return epoch_loss, epoch_metrics


def validate_epoch(model, dataloader, criterion, device, epoch):
    """Validate for one epoch."""
    model.eval()
    
    running_loss = 0.0
    running_metrics = {
        'dice': 0.0, 
        'jaccard': 0.0, 
        'iou': 0.0,
        'sensitivity': 0.0, 
        'specificity': 0.0, 
        'accuracy': 0.0
    }
    
    pbar = tqdm(dataloader, desc=f"Validation Epoch {epoch}")
    
    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            losses = criterion(outputs, masks)
            loss = losses['total']
            
            # Calculate metrics
            metrics = calculate_metrics(outputs['segmentation'], masks)
            
            # Update running stats
            running_loss += loss.item()
            for key in running_metrics:
                running_metrics[key] += metrics[key]
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'dice': f"{metrics['dice']:.4f}",
                'sens': f"{metrics['sensitivity']:.4f}"
            })
    
    # Calculate epoch averages
    num_batches = len(dataloader)
    epoch_loss = running_loss / num_batches
    epoch_metrics = {k: v / num_batches for k, v in running_metrics.items()}
    
    return epoch_loss, epoch_metrics


def save_checkpoint(model, optimizer, epoch, metrics, checkpoint_dir, is_best=False):
    """Save model checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    # Save regular checkpoint
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
    torch.save(checkpoint, checkpoint_path)
    
    # Save best checkpoint
    if is_best:
        best_path = checkpoint_dir / "best_model.pth"
        torch.save(checkpoint, best_path)
        print(f"✓ Saved best model to {best_path}")


def visualize_predictions(model, dataloader, device, save_dir, num_samples=5):
    """Visualize model predictions."""
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    samples_saved = 0
    
    with torch.no_grad():
        for images, masks in dataloader:
            if samples_saved >= num_samples:
                break
            
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            preds = torch.sigmoid(outputs['segmentation'])
            
            for i in range(images.shape[0]):
                if samples_saved >= num_samples:
                    break
                
                # Denormalize image
                img = images[i].cpu().numpy().transpose(1, 2, 0)
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img = np.clip(img, 0, 1)
                
                mask = masks[i, 0].cpu().numpy()
                pred = preds[i, 0].cpu().numpy()
                
                # Create visualization
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                axes[0].imshow(img)
                axes[0].set_title('Input Image')
                axes[0].axis('off')
                
                axes[1].imshow(mask, cmap='gray')
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')
                
                axes[2].imshow(pred, cmap='gray')
                axes[2].set_title('Prediction')
                axes[2].axis('off')
                
                plt.tight_layout()
                plt.savefig(save_dir / f"sample_{samples_saved}.png", dpi=150, bbox_inches='tight')
                plt.close()
                
                samples_saved += 1


def main():
    parser = argparse.ArgumentParser(description="Train MLMO-EMO Segmentation Model")
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='segmentation_data/train_valid',
                       help='Directory containing training data')
    parser.add_argument('--csv-path', type=str, default='unified_segmentation_dataset.csv',
                       help='Path to CSV file with image/mask pairs')
    parser.add_argument('--use-csv', action='store_true',
                       help='Use CSV file instead of directory structure')
    
    # Model arguments
    parser.add_argument('--encoder', type=str, default='resnet34',
                       choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 
                               'efficientnet-b0', 'efficientnet-b4'],
                       help='Encoder backbone')
    parser.add_argument('--num-particles', type=int, default=3,
                       help='Number of electromagnetic particles')
    parser.add_argument('--emo-iterations', type=int, default=3,
                       help='Number of EMO refinement iterations')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=150,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=12,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--img-size', type=int, default=512,
                       help='Input image size')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Loss weights
    parser.add_argument('--dice-weight', type=float, default=1.0)
    parser.add_argument('--bce-weight', type=float, default=1.0)
    parser.add_argument('--boundary-weight', type=float, default=0.5)
    parser.add_argument('--homogeneity-weight', type=float, default=0.3)
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_mlmo_emo',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--logdir', type=str, default='runs/mlmo_emo_segmentation',
                       help='TensorBoard log directory')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("\n" + "="*60)
    print("Loading Dataset")
    print("="*60)
    
    if args.use_csv and Path(args.csv_path).exists():
        train_imgs, val_imgs, train_masks, val_masks = load_dataset_from_csv(
            args.csv_path, args.data_dir
        )
    else:
        train_imgs, val_imgs, train_masks, val_masks = load_dataset_from_directory(
            args.data_dir
        )
    
    # Create datasets
    train_dataset = MammographyDataset(
        train_imgs, train_masks,
        transform=get_training_augmentation(args.img_size),
        img_size=args.img_size
    )
    
    val_dataset = MammographyDataset(
        val_imgs, val_masks,
        transform=get_validation_augmentation(args.img_size),
        img_size=args.img_size
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    print("\n" + "="*60)
    print("Creating MLMO-EMO Model")
    print("="*60)
    
    model = create_mlmo_emo_model(
        encoder_name=args.encoder,
        encoder_weights='imagenet',
        num_classes=1,
        num_particles=args.num_particles,
        emo_iterations=args.emo_iterations
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create loss function
    criterion = MLMOEMOLoss(
        dice_weight=args.dice_weight,
        bce_weight=args.bce_weight,
        boundary_weight=args.boundary_weight,
        homogeneity_weight=args.homogeneity_weight
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_dice = 0.0
    
    if args.resume and Path(args.resume).exists():
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint['metrics'].get('dice', 0.0)
        print(f"Resumed from epoch {start_epoch}, best dice: {best_dice:.4f}")
    
    # TensorBoard writer
    writer = SummaryWriter(args.logdir)
    
    # Training loop
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch+1
        )
        
        # Validate
        val_loss, val_metrics = validate_epoch(
            model, val_loader, criterion, device, epoch+1
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log to TensorBoard - Loss
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        
        # Log to TensorBoard - Dice Coefficient
        writer.add_scalar('Metrics/Dice/train', train_metrics['dice'], epoch)
        writer.add_scalar('Metrics/Dice/val', val_metrics['dice'], epoch)
        
        # Log to TensorBoard - Jaccard Coefficient (IoU)
        writer.add_scalar('Metrics/Jaccard/train', train_metrics['jaccard'], epoch)
        writer.add_scalar('Metrics/Jaccard/val', val_metrics['jaccard'], epoch)
        writer.add_scalar('Metrics/IoU/train', train_metrics['iou'], epoch)
        writer.add_scalar('Metrics/IoU/val', val_metrics['iou'], epoch)
        
        # Log to TensorBoard - Sensitivity (Recall/TPR)
        writer.add_scalar('Metrics/Sensitivity/train', train_metrics['sensitivity'], epoch)
        writer.add_scalar('Metrics/Sensitivity/val', val_metrics['sensitivity'], epoch)
        
        # Log to TensorBoard - Specificity (TNR)
        writer.add_scalar('Metrics/Specificity/train', train_metrics['specificity'], epoch)
        writer.add_scalar('Metrics/Specificity/val', val_metrics['specificity'], epoch)
        
        # Log to TensorBoard - Accuracy
        writer.add_scalar('Metrics/Accuracy/train', train_metrics['accuracy'], epoch)
        writer.add_scalar('Metrics/Accuracy/val', val_metrics['accuracy'], epoch)
        
        # Log to TensorBoard - Learning Rate
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        # Print epoch results
        print(f"\nTrain Loss: {train_loss:.4f} | Dice: {train_metrics['dice']:.4f} | "
              f"Jaccard: {train_metrics['jaccard']:.4f} | Sens: {train_metrics['sensitivity']:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Dice: {val_metrics['dice']:.4f} | "
              f"Jaccard: {val_metrics['jaccard']:.4f} | Sens: {val_metrics['sensitivity']:.4f}")
        print(f"Val Specificity: {val_metrics['specificity']:.4f} | Accuracy: {val_metrics['accuracy']:.4f}")
        
        # Save checkpoint
        is_best = val_metrics['dice'] > best_dice
        if is_best:
            best_dice = val_metrics['dice']
        
        save_checkpoint(
            model, optimizer, epoch, val_metrics,
            args.checkpoint_dir, is_best=is_best
        )
        
        # Visualize predictions periodically
        if (epoch + 1) % 10 == 0:
            vis_dir = Path(args.checkpoint_dir) / f"visualizations_epoch_{epoch+1}"
            visualize_predictions(model, val_loader, device, vis_dir, num_samples=5)
    
    writer.close()
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best validation Dice: {best_dice:.4f}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    print(f"TensorBoard logs: {args.logdir}")


if __name__ == "__main__":
    main()
