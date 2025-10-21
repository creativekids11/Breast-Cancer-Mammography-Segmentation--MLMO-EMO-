"""
Inference script for MLMO-EMO Breast Cancer Mammography Segmentation Model
"""

import os
import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

from mlmo_emo_segmentation import create_mlmo_emo_model


def get_inference_transform(img_size=512):
    """Get inference preprocessing pipeline."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def load_image(image_path, transform, device):
    """Load and preprocess a single image."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    original_shape = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply transform
    transformed = transform(image=image_rgb)
    image_tensor = transformed['image'].unsqueeze(0).to(device)
    
    return image_tensor, image_rgb, original_shape


def save_prediction(pred_mask, output_path, original_shape=None):
    """Save prediction mask."""
    if original_shape is not None:
        pred_mask = cv2.resize(pred_mask, (original_shape[1], original_shape[0]))
    
    # Convert to uint8
    pred_mask = (pred_mask * 255).astype(np.uint8)
    cv2.imwrite(str(output_path), pred_mask)


def visualize_result(image, pred_mask, save_path, original_shape=None):
    """Create and save visualization of prediction."""
    if original_shape is not None:
        pred_mask = cv2.resize(pred_mask, (original_shape[1], original_shape[0]))
    
    # Resize image to match mask if needed
    if image.shape[:2] != pred_mask.shape:
        image = cv2.resize(image, (pred_mask.shape[1], pred_mask.shape[0]))
    
    # Create colored overlay
    overlay = image.copy()
    mask_colored = np.zeros_like(image)
    mask_colored[:, :, 0] = pred_mask * 255  # Red channel for segmentation
    
    # Blend
    overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(pred_mask, cmap='gray')
    axes[1].set_title('Segmentation Mask', fontsize=14)
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay', fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def inference_single_image(model, image_path, output_dir, transform, device, 
                          threshold=0.5, visualize=True):
    """Run inference on a single image."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load image
    image_tensor, image_rgb, original_shape = load_image(image_path, transform, device)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        pred = torch.sigmoid(outputs['segmentation'])
        pred_mask = (pred[0, 0].cpu().numpy() > threshold).astype(np.float32)
    
    # Save results
    image_name = Path(image_path).stem
    
    # Save mask
    mask_path = output_dir / f"{image_name}_mask.png"
    save_prediction(pred_mask, mask_path, original_shape)
    
    # Save visualization
    if visualize:
        vis_path = output_dir / f"{image_name}_visualization.png"
        visualize_result(image_rgb, pred_mask, vis_path, original_shape)
    
    return pred_mask


def inference_batch(model, image_dir, output_dir, transform, device, 
                   threshold=0.5, visualize=True, extensions=('.png', '.jpg', '.jpeg')):
    """Run inference on a batch of images."""
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    image_paths = []
    for ext in extensions:
        image_paths.extend(list(image_dir.glob(f"*{ext}")))
        image_paths.extend(list(image_dir.glob(f"*{ext.upper()}")))
    
    if len(image_paths) == 0:
        print(f"No images found in {image_dir}")
        return
    
    print(f"Found {len(image_paths)} images")
    
    # Process each image
    for image_path in tqdm(image_paths, desc="Processing images"):
        try:
            inference_single_image(
                model, image_path, output_dir, transform, device,
                threshold, visualize
            )
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    print(f"\nInference complete! Results saved to: {output_dir}")


def calculate_metrics(pred_mask, gt_mask, threshold=0.5):
    """Calculate comprehensive segmentation metrics using confusion matrix."""
    pred_binary = (pred_mask > threshold).astype(np.float32)
    gt_binary = (gt_mask > 0).astype(np.float32)
    
    # Calculate confusion matrix components
    TP = ((pred_binary == 1) & (gt_binary == 1)).sum()
    TN = ((pred_binary == 0) & (gt_binary == 0)).sum()
    FP = ((pred_binary == 1) & (gt_binary == 0)).sum()
    FN = ((pred_binary == 0) & (gt_binary == 1)).sum()
    
    # Calculate metrics
    dice = (2.0 * TP + 1e-7) / (2.0 * TP + FP + FN + 1e-7)
    jaccard = (TP + 1e-7) / (TP + FP + FN + 1e-7)
    iou = jaccard  # Jaccard and IoU are the same
    sensitivity = (TP + 1e-7) / (TP + FN + 1e-7)  # Recall/TPR
    specificity = (TN + 1e-7) / (TN + FP + 1e-7)  # TNR
    accuracy = (TP + TN + 1e-7) / (TP + TN + FP + FN + 1e-7)
    
    return {
        'dice': float(dice),
        'jaccard': float(jaccard),
        'iou': float(iou),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'accuracy': float(accuracy)
    }


def evaluate_with_ground_truth(model, image_dir, mask_dir, transform, device, threshold=0.5):
    """Evaluate model with ground truth masks."""
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    
    # Get image paths
    image_paths = sorted(list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")))
    
    all_metrics = {
        'dice': [], 
        'jaccard': [],
        'iou': [], 
        'sensitivity': [],
        'specificity': [],
        'accuracy': []
    }
    
    print(f"Evaluating on {len(image_paths)} images with ground truth...")
    
    for image_path in tqdm(image_paths):
        # Find corresponding mask
        mask_path = mask_dir / image_path.name
        if not mask_path.exists():
            mask_path = mask_dir / (image_path.stem + ".png")
        if not mask_path.exists():
            print(f"Warning: No mask found for {image_path.name}")
            continue
        
        # Load image and mask
        image_tensor, _, original_shape = load_image(image_path, transform, device)
        gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            print(f"Warning: Could not load mask {mask_path}")
            continue
        
        # Run inference
        model.eval()
        with torch.no_grad():
            outputs = model(image_tensor)
            pred = torch.sigmoid(outputs['segmentation'])
            pred_mask = pred[0, 0].cpu().numpy()
        
        # Resize prediction to match ground truth
        pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]))
        
        # Calculate metrics
        metrics = calculate_metrics(pred_mask, gt_mask, threshold)
        for key in all_metrics:
            all_metrics[key].append(metrics[key])
    
    # Calculate average metrics
    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    std_metrics = {k: np.std(v) for k, v in all_metrics.items()}
    
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Dice Coefficient: {avg_metrics['dice']:.4f} ± {std_metrics['dice']:.4f}")
    print(f"Jaccard Index:    {avg_metrics['jaccard']:.4f} ± {std_metrics['jaccard']:.4f}")
    print(f"IoU:              {avg_metrics['iou']:.4f} ± {std_metrics['iou']:.4f}")
    print(f"Sensitivity:      {avg_metrics['sensitivity']:.4f} ± {std_metrics['sensitivity']:.4f}")
    print(f"Specificity:      {avg_metrics['specificity']:.4f} ± {std_metrics['specificity']:.4f}")
    print(f"Pixel Accuracy:   {avg_metrics['accuracy']:.4f} ± {std_metrics['accuracy']:.4f}")
    print("="*60)
    
    return avg_metrics, std_metrics


def main():
    parser = argparse.ArgumentParser(description="MLMO-EMO Segmentation Inference")
    
    # Model arguments
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to model weights')
    parser.add_argument('--encoder', type=str, default='resnet34',
                       help='Encoder backbone (must match training)')
    parser.add_argument('--num-particles', type=int, default=3,
                       help='Number of electromagnetic particles (must match training)')
    parser.add_argument('--emo-iterations', type=int, default=3,
                       help='Number of EMO iterations (must match training)')
    
    # Input/Output arguments
    parser.add_argument('--image', type=str, default=None,
                       help='Path to single image for inference')
    parser.add_argument('--image-dir', type=str, default=None,
                       help='Directory containing images for batch inference')
    parser.add_argument('--output-dir', type=str, default='predictions',
                       help='Directory to save predictions')
    
    # Evaluation arguments
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate with ground truth masks')
    parser.add_argument('--mask-dir', type=str, default=None,
                       help='Directory containing ground truth masks (for evaluation)')
    
    # Inference arguments
    parser.add_argument('--img-size', type=int, default=512,
                       help='Input image size')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for binary segmentation')
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='Create visualization images')
    parser.add_argument('--no-visualize', dest='visualize', action='store_false',
                       help='Do not create visualization images')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.image is None and args.image_dir is None:
        parser.error("Either --image or --image-dir must be specified")
    
    if args.evaluate and args.mask_dir is None:
        parser.error("--mask-dir must be specified when --evaluate is used")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("\n" + "="*60)
    print("Loading MLMO-EMO Model")
    print("="*60)
    
    model = create_mlmo_emo_model(
        encoder_name=args.encoder,
        encoder_weights=None,  # Don't load pretrained weights
        num_classes=1,
        num_particles=args.num_particles,
        emo_iterations=args.emo_iterations
    )
    
    # Load weights
    if not Path(args.weights).exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    
    checkpoint = torch.load(args.weights, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'metrics' in checkpoint:
            print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"Checkpoint metrics: {checkpoint['metrics']}")
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"✓ Loaded weights from {args.weights}")
    
    # Get transform
    transform = get_inference_transform(args.img_size)
    
    # Run inference
    print("\n" + "="*60)
    print("Running Inference")
    print("="*60)
    
    if args.evaluate and args.mask_dir:
        # Evaluation mode
        evaluate_with_ground_truth(
            model,
            args.image_dir if args.image_dir else Path(args.image).parent,
            args.mask_dir,
            transform,
            device,
            args.threshold
        )
    elif args.image:
        # Single image inference
        print(f"Processing single image: {args.image}")
        pred_mask = inference_single_image(
            model, args.image, args.output_dir, transform, device,
            args.threshold, args.visualize
        )
        print(f"\n✓ Prediction saved to: {args.output_dir}")
    else:
        # Batch inference
        inference_batch(
            model, args.image_dir, args.output_dir, transform, device,
            args.threshold, args.visualize
        )


if __name__ == "__main__":
    main()
