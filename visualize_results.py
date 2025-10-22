"""
Visualization script to debug segmentation results.
This helps identify why metrics are lower than paper reports.
"""

import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def load_results(results_path):
    """Load the results JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)

def visualize_sample_results(results_dir, num_samples=5):
    """
    Visualize sample segmentation results to debug issues.
    
    Shows:
    1. Original image
    2. Ground truth mask
    3. Predicted segmentation
    4. Overlay comparison
    """
    results_dir = Path(results_dir)
    segmentation_dir = results_dir / 'segmentations'
    
    # Get list of result files
    result_files = sorted(segmentation_dir.glob('*.npz'))[:num_samples]
    
    if not result_files:
        print("No segmentation results found!")
        return
    
    print(f"Visualizing {len(result_files)} sample results...")
    
    fig, axes = plt.subplots(len(result_files), 4, figsize=(16, 4*len(result_files)))
    
    if len(result_files) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, result_file in enumerate(result_files):
        try:
            # Load result
            data = np.load(result_file, allow_pickle=True)
            
            original = data['original_image']
            ground_truth = data['ground_truth_mask']
            segmented = data['segmented_image']
            metrics = data['metrics'].item()
            
            # Normalize for display
            original_display = cv2.normalize(original, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Ensure masks are binary
            ground_truth_display = (ground_truth > 0).astype(np.uint8) * 255
            segmented_display = (segmented > 0).astype(np.uint8) * 255
            
            # Create overlay
            overlay = cv2.cvtColor(original_display, cv2.COLOR_GRAY2BGR)
            overlay[:, :, 1] = np.maximum(overlay[:, :, 1], ground_truth_display)  # Green = GT
            overlay[:, :, 2] = np.maximum(overlay[:, :, 2], segmented_display)     # Red = Predicted
            
            # Plot
            axes[idx, 0].imshow(original_display, cmap='gray')
            axes[idx, 0].set_title(f'Original\n{result_file.stem}')
            axes[idx, 0].axis('off')
            
            axes[idx, 1].imshow(ground_truth_display, cmap='gray')
            axes[idx, 1].set_title(f'Ground Truth\nPixels: {np.sum(ground_truth > 0)}')
            axes[idx, 1].axis('off')
            
            axes[idx, 2].imshow(segmented_display, cmap='gray')
            axes[idx, 2].set_title(f'Predicted\nPixels: {np.sum(segmented > 0)}')
            axes[idx, 2].axis('off')
            
            axes[idx, 3].imshow(overlay)
            axes[idx, 3].set_title(f'Overlay (GT=Green, Pred=Red)\n'
                                   f'Dice: {metrics["dice_coefficient"]:.4f}\n'
                                   f'Jaccard: {metrics["jaccard_coefficient"]:.4f}')
            axes[idx, 3].axis('off')
            
            # Print metrics
            print(f"\n{result_file.stem}:")
            print(f"  Original shape: {original.shape}")
            print(f"  GT mask pixels: {np.sum(ground_truth > 0)} ({np.sum(ground_truth > 0) / ground_truth.size * 100:.2f}%)")
            print(f"  Predicted pixels: {np.sum(segmented > 0)} ({np.sum(segmented > 0) / segmented.size * 100:.2f}%)")
            print(f"  Metrics: Dice={metrics['dice_coefficient']:.4f}, "
                  f"Jaccard={metrics['jaccard_coefficient']:.4f}, "
                  f"Sens={metrics['sensitivity']:.4f}, "
                  f"Spec={metrics['specificity']:.4f}")
            
        except Exception as e:
            print(f"Error processing {result_file}: {e}")
            continue
    
    plt.tight_layout()
    plt.savefig(results_dir / 'visualization_debug.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {results_dir / 'visualization_debug.png'}")
    plt.show()

def analyze_metrics_distribution(results_path):
    """Analyze the distribution of metrics."""
    results = load_results(results_path)
    
    if not results.get('detailed_results'):
        print("No detailed results available.")
        return
    
    # Extract metrics
    metrics = {
        'dice': [],
        'jaccard': [],
        'sensitivity': [],
        'specificity': [],
        'accuracy': []
    }
    
    for result in results['detailed_results']:
        if result.get('metrics'):
            m = result['metrics']
            metrics['dice'].append(m['dice_coefficient'])
            metrics['jaccard'].append(m['jaccard_coefficient'])
            metrics['sensitivity'].append(m['sensitivity'])
            metrics['specificity'].append(m['specificity'])
            metrics['accuracy'].append(m['accuracy'])
    
    # Plot distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (metric_name, values) in enumerate(metrics.items()):
        if values:
            axes[idx].hist(values, bins=20, edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'{metric_name.capitalize()} Distribution')
            axes[idx].set_xlabel(metric_name.capitalize())
            axes[idx].set_ylabel('Frequency')
            axes[idx].axvline(np.mean(values), color='red', linestyle='--', 
                             label=f'Mean: {np.mean(values):.4f}')
            axes[idx].legend()
    
    axes[5].axis('off')  # Hide the last subplot
    
    plt.tight_layout()
    results_dir = Path(results_path).parent
    plt.savefig(results_dir / 'metrics_distribution.png', dpi=150, bbox_inches='tight')
    print(f"Metrics distribution saved to: {results_dir / 'metrics_distribution.png'}")
    plt.show()

def diagnose_common_issues(results_path):
    """Diagnose common issues causing low performance."""
    results = load_results(results_path)
    
    print("\n" + "="*60)
    print("DIAGNOSTIC REPORT")
    print("="*60)
    
    # Check success rate
    success_rate = results['success_rate']
    print(f"\n1. Success Rate: {success_rate*100:.1f}%")
    if success_rate < 0.9:
        print("   ⚠️  Low success rate - some images failed to process")
        print("   → Check for file loading errors or preprocessing failures")
    
    # Check aggregate metrics
    agg = results['aggregate_metrics']
    
    print(f"\n2. Jaccard Coefficient: {agg['jaccard_coefficient']['mean']:.4f}")
    if agg['jaccard_coefficient']['mean'] < 0.01:
        print("   ⚠️  CRITICAL: Very low Jaccard (<1%)")
        print("   → Possible issues:")
        print("      - Predicted mask is mostly empty or wrong")
        print("      - Ground truth mask not loaded correctly")
        print("      - Threshold application error")
        print("      - Post-processing missing")
    
    print(f"\n3. Dice Coefficient: {agg['dice_coefficient']['mean']:.4f}")
    if agg['dice_coefficient']['mean'] < 0.02:
        print("   ⚠️  CRITICAL: Very low Dice (<2%)")
        print("   → Predicted and ground truth have minimal overlap")
    
    print(f"\n4. Sensitivity: {agg['sensitivity']['mean']*100:.2f}%")
    if agg['sensitivity']['mean'] < 0.7:
        print("   ⚠️  Low sensitivity")
        print("   → Model is missing many true positive pixels")
        print("   → Check if thresholds are too conservative")
    
    print(f"\n5. Specificity: {agg['specificity']['mean']*100:.2f}%")
    if agg['specificity']['mean'] < 0.9:
        print("   ⚠️  Low specificity")
        print("   → Too many false positives")
        print("   → Check if thresholds are too aggressive")
    
    print(f"\n6. Accuracy: {agg['accuracy']['mean']*100:.2f}%")
    if agg['accuracy']['mean'] < 0.9:
        print("   ⚠️  Low accuracy")
        print("   → Overall poor performance")
    
    # Check detailed results
    if results.get('detailed_results'):
        failed_count = sum(1 for r in results['detailed_results'] if r.get('error'))
        print(f"\n7. Failed Images: {failed_count}/{len(results['detailed_results'])}")
        if failed_count > 0:
            print("   ⚠️  Some images failed during processing")
            print("   → Check error messages in detailed results")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS:")
    print("="*60)
    print("\n1. Check preprocessing:")
    print("   - Verify image normalization matches paper")
    print("   - Confirm contrast enhancement parameters")
    
    print("\n2. Verify threshold application:")
    print("   - Check if optimized thresholds create proper binary masks")
    print("   - Ensure threshold range is appropriate (0-255)")
    
    print("\n3. Validate ground truth masks:")
    print("   - Confirm masks are loaded correctly")
    print("   - Check mask resolution matches image resolution")
    print("   - Verify mask pixel values (should be binary 0/1 or 0/255)")
    
    print("\n4. Add post-processing:")
    print("   - Morphological operations (opening, closing)")
    print("   - Hole filling")
    print("   - Connected component analysis")
    
    print("\n5. Review paper methodology:")
    print("   - Re-check preprocessing steps")
    print("   - Verify EML parameter settings")
    print("   - Confirm evaluation metric calculations")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    # Find the most recent results file
    results_dir = Path("paper_results")
    
    if not results_dir.exists():
        print("Error: paper_results directory not found!")
        sys.exit(1)
    
    result_files = sorted(results_dir.glob("mlmo_emo_results_*.json"))
    
    if not result_files:
        print("Error: No results files found!")
        sys.exit(1)
    
    latest_results = result_files[-1]
    print(f"Analyzing: {latest_results}")
    
    # Run diagnostics
    diagnose_common_issues(latest_results)
    
    # Visualize sample results
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS...")
    print("="*60)
    visualize_sample_results(results_dir, num_samples=5)
    
    # Analyze metrics distribution
    analyze_metrics_distribution(latest_results)
    
    print("\n✅ Analysis complete!")
    print(f"   Results: {latest_results}")
    print(f"   Visualizations: {results_dir / 'visualization_debug.png'}")
    print(f"   Metrics plot: {results_dir / 'metrics_distribution.png'}")
