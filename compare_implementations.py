"""
Comparison script between the original MLMO-EMO implementation and 
the exact paper implementation.

This script demonstrates the key differences and allows for side-by-side
comparison of both approaches.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, Tuple
import sys
import os

# Add parent directory to path for original implementation
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from .paper_exact_mlmo_emo import (
    PaperSegmentationModel, 
    PaperEvaluationMetrics,
    PaperPreprocessor
)

# Try to import original implementation from parent directory
try:
    from mlmo_emo_segmentation import create_mlmo_emo_model, MLMOEMOLoss
    import torch
    ORIGINAL_AVAILABLE = True
except ImportError:
    ORIGINAL_AVAILABLE = False
    print("Warning: Original MLMO-EMO implementation not available")


class ImplementationComparison:
    """
    Compare original deep learning implementation with exact paper implementation.
    """
    
    def __init__(self):
        """Initialize both implementations."""
        # Paper implementation (always available)
        self.paper_model = PaperSegmentationModel()
        self.paper_metrics = PaperEvaluationMetrics()
        
        # Original implementation (if available)
        if ORIGINAL_AVAILABLE:
            self.original_model = create_mlmo_emo_model(
                encoder_name="resnet18",
                num_classes=1,
                num_particles=2,
                emo_iterations=2,
                hidden_dim=64
            )
            self.original_loss = MLMOEMOLoss()
            self.original_model.eval()  # Set to evaluation mode
        else:
            self.original_model = None
            self.original_loss = None
    
    def create_test_mammogram(self, size: Tuple[int, int] = (512, 512)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a synthetic mammogram for testing purposes.
        
        Returns:
            Tuple of (image, ground_truth_mask)
        """
        height, width = size
        
        # Create base image with some texture
        image = np.random.randint(50, 200, (height, width), dtype=np.uint8)
        
        # Add breast tissue structure
        center_x, center_y = width // 3, height // 2
        
        # Create circular and irregular masses
        y, x = np.ogrid[:height, :width]
        
        # Mass 1 - circular
        mask1 = (x - center_x)**2 + (y - center_y)**2 < (80)**2
        
        # Mass 2 - irregular shape
        mask2 = ((x - (center_x + 150))**2 / (60)**2 + 
                (y - (center_y - 50))**2 / (40)**2) < 1
        
        # Add masses to image
        image[mask1] = np.clip(image[mask1] + 60, 0, 255)
        image[mask2] = np.clip(image[mask2] + 45, 0, 255)
        
        # Create ground truth (both masses)
        ground_truth = np.zeros_like(image)
        ground_truth[mask1 | mask2] = 255
        
        return image, ground_truth
    
    def run_paper_implementation(self, image: np.ndarray, 
                               method: str = 'otsu') -> Dict:
        """
        Run the exact paper implementation.
        
        Args:
            image: Input mammogram
            method: 'otsu' or 'kapur'
            
        Returns:
            Results dictionary
        """
        start_time = time.time()
        
        # Segmentation using paper method
        segmentation_results = self.paper_model.segment_image(
            image, method=method, num_thresholds=1
        )
        
        processing_time = time.time() - start_time
        
        return {
            'method': 'Paper Implementation',
            'algorithm': f'EML + {method.upper()}',
            'segmented': segmentation_results['segmented'],
            'preprocessing_steps': segmentation_results['preprocessing_steps'],
            'thresholds': segmentation_results['thresholds'],
            'processing_time': processing_time
        }
    
    def run_original_implementation(self, image: np.ndarray) -> Dict:
        """
        Run the original deep learning implementation.
        
        Args:
            image: Input mammogram
            
        Returns:
            Results dictionary
        """
        if not ORIGINAL_AVAILABLE:
            return {
                'method': 'Original Implementation',
                'algorithm': 'Deep Learning MLMO-EMO',
                'segmented': None,
                'error': 'Original implementation not available'
            }
        
        start_time = time.time()
        
        try:
            # Prepare input for PyTorch model
            # Convert grayscale to RGB
            if len(image.shape) == 2:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = image
            
            # Resize and normalize
            image_resized = cv2.resize(image_rgb, (512, 512))
            image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float()
            image_tensor = image_tensor / 255.0  # Normalize to [0, 1]
            
            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0)
            
            # Run inference
            with torch.no_grad():
                outputs = self.original_model(image_tensor)
            
            # Get segmentation output
            segmentation = outputs['segmentation']
            segmentation = torch.sigmoid(segmentation)
            segmentation = (segmentation > 0.5).float()
            
            # Convert back to numpy
            segmented_np = segmentation.squeeze().cpu().numpy()
            segmented_np = (segmented_np * 255).astype(np.uint8)
            
            # Resize back to original size if needed
            if segmented_np.shape != image.shape[:2]:
                segmented_np = cv2.resize(segmented_np, 
                                       (image.shape[1], image.shape[0]))
            
            processing_time = time.time() - start_time
            
            return {
                'method': 'Original Implementation',
                'algorithm': 'Deep Learning MLMO-EMO',
                'segmented': segmented_np,
                'model_outputs': outputs,
                'processing_time': processing_time
            }
            
        except Exception as e:
            return {
                'method': 'Original Implementation',
                'algorithm': 'Deep Learning MLMO-EMO',
                'segmented': None,
                'error': str(e)
            }
    
    def compare_implementations(self, image: np.ndarray, 
                              ground_truth: np.ndarray) -> Dict:
        """
        Compare both implementations side by side.
        
        Args:
            image: Input mammogram
            ground_truth: Ground truth segmentation
            
        Returns:
            Comparison results
        """
        print("Running implementation comparison...")
        
        # Run paper implementation
        print("1. Running paper implementation (OTSU)...")
        paper_results_otsu = self.run_paper_implementation(image, 'otsu')
        
        print("2. Running paper implementation (Kapur)...")
        paper_results_kapur = self.run_paper_implementation(image, 'kapur')
        
        # Run original implementation
        print("3. Running original implementation...")
        original_results = self.run_original_implementation(image)
        
        # Calculate metrics for all methods
        results = {}
        
        # Paper OTSU metrics
        if paper_results_otsu['segmented'] is not None:
            paper_otsu_metrics = self.paper_metrics.calculate_metrics(
                paper_results_otsu['segmented'], ground_truth
            )
            results['paper_otsu'] = {
                **paper_results_otsu,
                'metrics': paper_otsu_metrics
            }
        
        # Paper Kapur metrics  
        if paper_results_kapur['segmented'] is not None:
            paper_kapur_metrics = self.paper_metrics.calculate_metrics(
                paper_results_kapur['segmented'], ground_truth
            )
            results['paper_kapur'] = {
                **paper_results_kapur,
                'metrics': paper_kapur_metrics
            }
        
        # Original metrics
        if original_results['segmented'] is not None:
            original_metrics = self.paper_metrics.calculate_metrics(
                original_results['segmented'], ground_truth
            )
            results['original'] = {
                **original_results,
                'metrics': original_metrics
            }
        else:
            results['original'] = original_results
        
        return results
    
    def visualize_comparison(self, image: np.ndarray, ground_truth: np.ndarray,
                           results: Dict):
        """
        Create visualization comparing all implementations.
        
        Args:
            image: Original image
            ground_truth: Ground truth mask
            results: Comparison results
        """
        # Count available results
        available_results = [k for k, v in results.items() 
                           if v.get('segmented') is not None]
        
        n_results = len(available_results)
        n_cols = min(4, n_results + 2)  # +2 for original and ground truth
        n_rows = max(2, (n_results + 3) // 4)  # Adaptive rows
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Original image
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Mammogram')
        axes[0, 0].axis('off')
        
        # Ground truth
        axes[0, 1].imshow(ground_truth, cmap='gray')
        axes[0, 1].set_title('Ground Truth')
        axes[0, 1].axis('off')
        
        # Plot results
        col = 2
        row = 0
        
        for key in available_results:
            if col >= n_cols:
                col = 0
                row += 1
                
            result = results[key]
            segmented = result['segmented']
            
            axes[row, col].imshow(segmented, cmap='gray')
            
            # Create title with method and metrics
            method_name = result.get('method', key)
            algorithm = result.get('algorithm', '')
            
            title = f"{method_name}\n{algorithm}"
            
            if 'metrics' in result:
                metrics = result['metrics']
                title += f"\nAcc: {metrics['accuracy']:.3f}"
                title += f", Dice: {metrics['dice_coefficient']:.3f}"
            
            if 'processing_time' in result:
                title += f"\nTime: {result['processing_time']:.2f}s"
            
            axes[row, col].set_title(title, fontsize=10)
            axes[row, col].axis('off')
            
            col += 1
        
        # Hide remaining subplots
        total_plots = n_rows * n_cols
        used_plots = len(available_results) + 2
        
        for i in range(used_plots, total_plots):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.suptitle('MLMO-EMO Implementation Comparison', 
                    fontsize=16, y=1.02)
        plt.show()
        
        return fig
    
    def print_comparison_summary(self, results: Dict):
        """
        Print detailed comparison summary.
        
        Args:
            results: Comparison results
        """
        print("\n" + "="*80)
        print("IMPLEMENTATION COMPARISON SUMMARY")
        print("="*80)
        
        # Print method comparison table
        print("\nMETHOD COMPARISON:")
        print("-" * 80)
        print(f"{'Method':<25} {'Algorithm':<25} {'Time (s)':<10} {'Accuracy':<10} {'Dice':<10}")
        print("-" * 80)
        
        for key, result in results.items():
            method = result.get('method', key)[:24]
            algorithm = result.get('algorithm', 'N/A')[:24]
            
            if 'processing_time' in result:
                time_str = f"{result['processing_time']:.2f}"
            else:
                time_str = "N/A"
            
            if 'metrics' in result:
                metrics = result['metrics']
                acc_str = f"{metrics['accuracy']:.3f}"
                dice_str = f"{metrics['dice_coefficient']:.3f}"
            else:
                acc_str = "N/A"
                dice_str = "N/A"
            
            print(f"{method:<25} {algorithm:<25} {time_str:<10} {acc_str:<10} {dice_str:<10}")
        
        print("-" * 80)
        
        # Detailed metrics for each method
        for key, result in results.items():
            if 'metrics' in result:
                print(f"\nDETAILED METRICS - {result.get('method', key)}:")
                print("-" * 40)
                metrics = result['metrics']
                
                print(f"Jaccard Coefficient: {metrics['jaccard_coefficient']:.4f}")
                print(f"Dice Coefficient: {metrics['dice_coefficient']:.4f}")
                print(f"Sensitivity: {metrics['sensitivity']:.4f}")
                print(f"Specificity: {metrics['specificity']:.4f}")
                print(f"Accuracy: {metrics['accuracy']:.4f}")
                
                if 'thresholds' in result:
                    print(f"Optimal Thresholds: {result['thresholds']}")
        
        print("\n" + "="*80)
        print("KEY DIFFERENCES:")
        print("-" * 40)
        print("Paper Implementation:")
        print("  ✓ Exact methodology from paper")
        print("  ✓ Classical computer vision approach")
        print("  ✓ Electromagnetism-like optimization")
        print("  ✓ OTSU/Kapur thresholding")
        print("  ✓ No training required")
        print("  - Slower processing")
        
        if ORIGINAL_AVAILABLE:
            print("\nOriginal Implementation:")
            print("  ✓ Deep learning approach")
            print("  ✓ End-to-end learning")
            print("  ✓ Faster inference")
            print("  ✓ Modern PyTorch architecture")
            print("  - Requires training")
            print("  - Different from paper methodology")
        else:
            print("\nOriginal Implementation:")
            print("  ✗ Not available (missing dependencies)")
        
        print("="*80)


def main():
    """Main comparison demonstration."""
    print("MLMO-EMO Implementation Comparison")
    print("=" * 50)
    
    # Initialize comparison
    comparison = ImplementationComparison()
    
    # Create test data
    print("Creating synthetic test mammogram...")
    image, ground_truth = comparison.create_test_mammogram()
    
    print(f"Image size: {image.shape}")
    print(f"Ground truth size: {ground_truth.shape}")
    
    # Run comparison
    results = comparison.compare_implementations(image, ground_truth)
    
    # Print summary
    comparison.print_comparison_summary(results)
    
    # Create visualization
    print("\nGenerating comparison visualization...")
    try:
        fig = comparison.visualize_comparison(image, ground_truth, results)
        print("Visualization created successfully!")
    except Exception as e:
        print(f"Visualization error: {e}")
    
    return results


if __name__ == "__main__":
    results = main()