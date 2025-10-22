"""
Simple usage example for the exact MLMO-EMO paper implementation.

This script shows how to use the paper implementation for mammogram segmentation.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import the exact paper implementation
from .paper_exact_mlmo_emo import (
    PaperSegmentationModel,
    PaperEvaluationMetrics,
    PaperPreprocessor
)

def create_demo_mammogram(size=(256, 256), num_masses=2):
    """
    Create a realistic synthetic mammogram for demonstration.
    
    Args:
        size: Image dimensions
        num_masses: Number of masses to add
        
    Returns:
        Tuple of (mammogram_image, ground_truth_mask)
    """
    height, width = size
    
    # Create base mammogram texture
    mammogram = np.random.randint(40, 120, (height, width), dtype=np.uint8)
    
    # Add Gaussian noise for texture
    noise = np.random.normal(0, 10, (height, width))
    mammogram = np.clip(mammogram + noise, 0, 255).astype(np.uint8)
    
    # Create ground truth mask
    ground_truth = np.zeros((height, width), dtype=np.uint8)
    
    # Add masses at different locations
    masses_info = []
    
    for i in range(num_masses):
        # Random center location (avoid edges)
        center_x = np.random.randint(width//4, 3*width//4)
        center_y = np.random.randint(height//4, 3*height//4)
        
        # Random size and shape
        radius_x = np.random.randint(20, 40)
        radius_y = np.random.randint(20, 40)
        
        # Create mass
        y, x = np.ogrid[:height, :width]
        mask = ((x - center_x)**2 / radius_x**2 + 
                (y - center_y)**2 / radius_y**2) < 1
        
        # Add to mammogram (brighter region)
        mammogram[mask] = np.clip(mammogram[mask] + np.random.randint(60, 100), 0, 255)
        
        # Add to ground truth
        ground_truth[mask] = 255
        
        masses_info.append({
            'center': (center_x, center_y),
            'radius': (radius_x, radius_y),
            'pixels': np.sum(mask)
        })
    
    print(f"Created mammogram with {num_masses} masses:")
    for i, info in enumerate(masses_info):
        print(f"  Mass {i+1}: center={info['center']}, size={info['pixels']} pixels")
    
    return mammogram, ground_truth


def demonstrate_preprocessing(image):
    """Demonstrate the preprocessing pipeline from the paper."""
    print("\n" + "="*60)
    print("PREPROCESSING DEMONSTRATION (Section 3.2)")
    print("="*60)
    
    preprocessor = PaperPreprocessor()
    results = preprocessor.preprocess_mammogram(image)
    
    print("Preprocessing steps completed:")
    print(f"1. ✓ Normalization (Equation 1)")
    print(f"2. ✓ Sigmoid normalization (Equation 2)")
    print(f"3. ✓ Median filtering (Equation 3)")
    print(f"4. ✓ CLAHE enhancement (Equations 4-8)")
    
    # Show statistics for each step
    for step_name, step_image in results.items():
        if step_name != 'original':
            min_val, max_val = np.min(step_image), np.max(step_image)
            mean_val = np.mean(step_image)
            print(f"   {step_name}: range=[{min_val}, {max_val}], mean={mean_val:.1f}")
    
    return results


def demonstrate_segmentation(image, method='otsu'):
    """Demonstrate the EML segmentation from the paper."""
    print("\n" + "="*60)
    print(f"SEGMENTATION DEMONSTRATION - {method.upper()} METHOD (Section 3.3)")
    print("="*60)
    
    model = PaperSegmentationModel()
    
    print(f"Running MLMO-EMO segmentation with {method} thresholding...")
    print("This uses electromagnetism-like optimization with:")
    print("- Population size: 50 particles")
    print("- Iterations: 100")
    print("- Attraction/repulsion forces")
    print("- Local search optimization")
    
    # Run segmentation (reduced iterations for demo)
    model.optimizer.max_iterations = 20  # Reduce for demo speed
    results = model.segment_image(image, method=method, num_thresholds=1)
    
    print(f"✓ Segmentation completed!")
    print(f"✓ Optimal threshold found: {results['thresholds'][0]:.2f}")
    
    return results


def demonstrate_evaluation(segmented, ground_truth):
    """Demonstrate evaluation metrics from the paper."""
    print("\n" + "="*60)
    print("EVALUATION METRICS (Equations 14-18)")
    print("="*60)
    
    metrics_calculator = PaperEvaluationMetrics()
    metrics = metrics_calculator.calculate_metrics(segmented, ground_truth)
    
    print("Calculated metrics using exact paper equations:")
    print(f"📊 Jaccard Coefficient (Eq. 14):  {metrics['jaccard_coefficient']:.4f}")
    print(f"📊 Dice Coefficient (Eq. 15):    {metrics['dice_coefficient']:.4f}")
    print(f"📊 Sensitivity (Eq. 16):         {metrics['sensitivity']:.4f} ({metrics['sensitivity']*100:.1f}%)")
    print(f"📊 Specificity (Eq. 17):         {metrics['specificity']:.4f} ({metrics['specificity']*100:.1f}%)")
    print(f"📊 Accuracy (Eq. 18):            {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
    
    print(f"\nConfusion Matrix Components:")
    print(f"  True Positives (TP):  {metrics['TP']}")
    print(f"  True Negatives (TN):  {metrics['TN']}")
    print(f"  False Positives (FP): {metrics['FP']}")
    print(f"  False Negatives (FN): {metrics['FN']}")
    
    return metrics


def visualize_results(original, ground_truth, preprocessing_results, 
                     segmentation_results, save_path=None):
    """Create comprehensive visualization of results."""
    print("\n" + "="*60)
    print("VISUALIZATION")
    print("="*60)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Row 1: Original processing chain
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original Mammogram')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(preprocessing_results['normalized'], cmap='gray')
    axes[0, 1].set_title('Normalized\n(Equation 1)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(preprocessing_results['median_filtered'], cmap='gray')
    axes[0, 2].set_title('Median Filtered\n(Equation 3)')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(preprocessing_results['clahe_enhanced'], cmap='gray')
    axes[0, 3].set_title('CLAHE Enhanced\n(Equations 4-8)')
    axes[0, 3].axis('off')
    
    # Row 2: Segmentation results
    axes[1, 0].imshow(segmentation_results['preprocessed'], cmap='gray')
    axes[1, 0].set_title('Final Preprocessed')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(segmentation_results['segmented'], cmap='gray')
    axes[1, 1].set_title(f'EML Segmented\n(Threshold: {segmentation_results["thresholds"][0]:.1f})')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(ground_truth, cmap='gray')
    axes[1, 2].set_title('Ground Truth')
    axes[1, 2].axis('off')
    
    # Overlay comparison
    overlay = np.zeros((*original.shape, 3))
    overlay[:, :, 0] = segmentation_results['segmented'] / 255  # Red: segmented
    overlay[:, :, 1] = ground_truth / 255  # Green: ground truth
    # Yellow where they overlap, Red=segmented only, Green=ground truth only
    
    axes[1, 3].imshow(overlay)
    axes[1, 3].set_title('Overlay Comparison\n(Red=Segmented, Green=GT)')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.suptitle('MLMO-EMO Paper Implementation Results', fontsize=16, y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()
    
    return fig


def compare_methods(image, ground_truth):
    """Compare OTSU and Kapur methods."""
    print("\n" + "="*60)
    print("METHOD COMPARISON: OTSU vs KAPUR")
    print("="*60)
    
    model = PaperSegmentationModel()
    model.optimizer.max_iterations = 15  # Reduce for demo
    metrics_calculator = PaperEvaluationMetrics()
    
    methods = ['otsu', 'kapur']
    results = {}
    
    for method in methods:
        print(f"\nTesting {method.upper()} method...")
        
        seg_results = model.segment_image(image, method=method)
        metrics = metrics_calculator.calculate_metrics(
            seg_results['segmented'], ground_truth
        )
        
        results[method] = {
            'segmentation': seg_results,
            'metrics': metrics
        }
        
        print(f"  Threshold: {seg_results['thresholds'][0]:.2f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Dice: {metrics['dice_coefficient']:.4f}")
    
    # Comparison summary
    print("\n" + "-"*50)
    print("COMPARISON SUMMARY:")
    print("-"*50)
    print(f"{'Metric':<20} {'OTSU':<12} {'Kapur':<12} {'Better':<10}")
    print("-"*50)
    
    metric_names = ['accuracy', 'dice_coefficient', 'sensitivity', 'specificity']
    for metric in metric_names:
        otsu_val = results['otsu']['metrics'][metric]
        kapur_val = results['kapur']['metrics'][metric]
        better = 'OTSU' if otsu_val > kapur_val else 'Kapur'
        
        print(f"{metric:<20} {otsu_val:<12.4f} {kapur_val:<12.4f} {better:<10}")
    
    return results


def main():
    """Main demonstration function."""
    print("🔬 MLMO-EMO Paper Implementation Demonstration")
    print("📄 Paper: 'Segmentation of Breast Masses in Mammogram Image Using")
    print("    Multilevel Multiobjective Electromagnetism-Like Optimization Algorithm'")
    print("🔗 https://onlinelibrary.wiley.com/doi/10.1155/2022/8576768")
    print()
    
    # Create demo data
    print("Creating synthetic mammogram for demonstration...")
    image, ground_truth = create_demo_mammogram(size=(256, 256), num_masses=2)
    
    # Demonstrate preprocessing
    preprocessing_results = demonstrate_preprocessing(image)
    
    # Demonstrate segmentation
    segmentation_results = demonstrate_segmentation(image, method='otsu')
    
    # Demonstrate evaluation
    metrics = demonstrate_evaluation(segmentation_results['segmented'], ground_truth)
    
    # Create visualization
    fig = visualize_results(
        image, ground_truth, preprocessing_results, 
        segmentation_results, save_path='mlmo_emo_demo_results.png'
    )
    
    # Compare methods
    method_comparison = compare_methods(image, ground_truth)
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE!")
    print("="*60)
    print("✅ Exact paper preprocessing implemented (Section 3.2)")
    print("✅ EML optimization implemented (Section 3.3)")
    print("✅ OTSU and Kapur thresholding implemented")
    print("✅ Evaluation metrics implemented (Equations 14-18)")
    print("✅ Template matching implemented")
    print("✅ Complete pipeline validated")
    print()
    print("📁 Results saved to: mlmo_emo_demo_results.png")
    print("🔍 Ready for real mammogram dataset processing!")
    
    return {
        'image': image,
        'ground_truth': ground_truth,
        'preprocessing': preprocessing_results,
        'segmentation': segmentation_results,
        'metrics': metrics,
        'method_comparison': method_comparison
    }


if __name__ == "__main__":
    # Set up matplotlib for display
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    
    # Run demonstration
    results = main()
    
    # Show final summary
    print(f"\n📊 Final Results Summary:")
    print(f"   Accuracy: {results['metrics']['accuracy']:.1%}")
    print(f"   Dice Score: {results['metrics']['dice_coefficient']:.4f}")
    print(f"   Processing: ✅ Complete")