"""
Training script using the exact MLMO-EMO implementation from the paper.

This script integrates with the existing dataset processing but uses the 
paper's exact methodology for segmentation.
"""

import os
import argparse
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import json
from datetime import datetime

# Import the exact paper implementation
try:
    # Try relative import (when run as module)
    from paper_exact_mlmo_emo import (
        PaperSegmentationModel, 
        PaperEvaluationMetrics,
        PaperPreprocessor
    )
except ImportError:
    # Try absolute import (when run directly)
    from paper_exact_mlmo_emo import (
        PaperSegmentationModel, 
        PaperEvaluationMetrics,
        PaperPreprocessor
    )


class PaperDatasetProcessor:
    """
    Dataset processor that works with the exact paper methodology.
    """
    
    def __init__(self, dataset_csv: str):
        """
        Initialize with dataset CSV file.
        
        Args:
            dataset_csv: Path to CSV file with image and mask paths
        """
        self.df = pd.read_csv(dataset_csv)
        self.model = PaperSegmentationModel()
        self.metrics_calculator = PaperEvaluationMetrics()
        
    def process_single_image(self, image_path: str, mask_path: str, 
                           method: str = 'otsu') -> Dict:
        """
        Process a single mammogram image using paper methodology.
        
        Args:
            image_path: Path to mammogram image
            mask_path: Path to ground truth mask
            method: 'otsu' or 'kapur' thresholding method
            
        Returns:
            Processing results including metrics
        """
        try:
            # Load image and mask
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
                
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Could not load mask: {mask_path}")
            
            # Apply paper's segmentation methodology
            segmentation_results = self.model.segment_image(
                image, method=method, num_thresholds=1
            )
            
            # Template matching (as described in paper)
            matching_results = self.model.template_matching(
                segmentation_results['segmented'], 
                mask
            )
            
            # Calculate exact evaluation metrics (equations 14-18)
            metrics = self.metrics_calculator.calculate_metrics(
                segmentation_results['segmented'], 
                mask
            )
            
            return {
                'image_path': image_path,
                'mask_path': mask_path,
                'segmentation_results': segmentation_results,
                'template_matching': matching_results,
                'metrics': metrics,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            return {
                'image_path': image_path,
                'mask_path': mask_path,
                'success': False,
                'error': str(e),
                'metrics': None
            }
    
    def process_dataset(self, method: str = 'otsu', 
                       max_images: int = None,
                       save_results: bool = True) -> Dict:
        """
        Process entire dataset using paper methodology.
        
        Args:
            method: 'otsu' or 'kapur' thresholding method
            max_images: Maximum number of images to process (None for all)
            save_results: Whether to save detailed results
            
        Returns:
            Aggregated results and statistics
        """
        print(f"Processing dataset with {method} method...")
        print(f"Total images in dataset: {len(self.df)}")
        
        # Limit number of images if specified
        df_subset = self.df.head(max_images) if max_images else self.df
        
        results = []
        all_metrics = []
        
        # Process each image with progress bar
        for idx, row in tqdm(df_subset.iterrows(), 
                           total=len(df_subset),
                           desc="Processing images"):
            
            image_path = row['image_path']
            mask_path = row['mask_path']
            
            # Process single image
            result = self.process_single_image(image_path, mask_path, method)
            results.append(result)
            
            # Collect metrics for successful processing
            if result['success'] and result['metrics']:
                all_metrics.append(result['metrics'])
            
            # Print progress every 10 images
            if (idx + 1) % 10 == 0:
                success_rate = sum(1 for r in results if r['success']) / len(results)
                print(f"Processed {idx + 1} images, success rate: {success_rate:.2%}")
        
        # Calculate aggregate statistics
        aggregate_stats = self._calculate_aggregate_statistics(all_metrics)
        
        # Prepare final results
        final_results = {
            'method': method,
            'total_images': len(df_subset),
            'successful_processing': len(all_metrics),
            'success_rate': len(all_metrics) / len(df_subset) if df_subset else 0,
            'aggregate_metrics': aggregate_stats,
            'detailed_results': results if save_results else None,
            'processing_timestamp': datetime.now().isoformat()
        }
        
        return final_results
    
    def _calculate_aggregate_statistics(self, metrics_list: List[Dict]) -> Dict:
        """
        Calculate aggregate statistics as reported in the paper.
        
        The paper reports average performance across datasets:
        - DDSM: 92.3% sensitivity, 99.21% specificity, 98.68% accuracy
        - MIAS: 92.11% sensitivity, 99.45% specificity, 98.93% accuracy
        """
        if not metrics_list:
            return {}
        
        # Extract all metric values
        jaccard_values = [m['jaccard_coefficient'] for m in metrics_list]
        dice_values = [m['dice_coefficient'] for m in metrics_list]
        sensitivity_values = [m['sensitivity'] for m in metrics_list]
        specificity_values = [m['specificity'] for m in metrics_list]
        accuracy_values = [m['accuracy'] for m in metrics_list]
        
        # Calculate statistics
        return {
            'jaccard_coefficient': {
                'mean': np.mean(jaccard_values),
                'std': np.std(jaccard_values),
                'min': np.min(jaccard_values),
                'max': np.max(jaccard_values)
            },
            'dice_coefficient': {
                'mean': np.mean(dice_values),
                'std': np.std(dice_values),
                'min': np.min(dice_values),
                'max': np.max(dice_values)
            },
            'sensitivity': {
                'mean': np.mean(sensitivity_values) * 100,  # Convert to percentage
                'std': np.std(sensitivity_values) * 100,
                'min': np.min(sensitivity_values) * 100,
                'max': np.max(sensitivity_values) * 100
            },
            'specificity': {
                'mean': np.mean(specificity_values) * 100,  # Convert to percentage
                'std': np.std(specificity_values) * 100,
                'min': np.min(specificity_values) * 100,
                'max': np.max(specificity_values) * 100
            },
            'accuracy': {
                'mean': np.mean(accuracy_values) * 100,  # Convert to percentage
                'std': np.std(accuracy_values) * 100,
                'min': np.min(accuracy_values) * 100,
                'max': np.max(accuracy_values) * 100
            }
        }
    
    def save_results(self, results: Dict, output_path: str):
        """Save results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to: {output_path}")
    
    def print_summary(self, results: Dict):
        """Print summary of results in paper format."""
        print("\n" + "="*60)
        print("MLMO-EMO SEGMENTATION RESULTS")
        print("="*60)
        
        print(f"Method: {results['method'].upper()}")
        print(f"Total images processed: {results['total_images']}")
        print(f"Successful processing: {results['successful_processing']}")
        print(f"Success rate: {results['success_rate']:.2%}")
        
        if results['aggregate_metrics']:
            print("\nAGGREGATE METRICS:")
            print("-" * 40)
            
            metrics = results['aggregate_metrics']
            
            print(f"Jaccard Coefficient: {metrics['jaccard_coefficient']['mean']:.4f} ± "
                  f"{metrics['jaccard_coefficient']['std']:.4f}")
            
            print(f"Dice Coefficient: {metrics['dice_coefficient']['mean']:.4f} ± "
                  f"{metrics['dice_coefficient']['std']:.4f}")
            
            print(f"Sensitivity: {metrics['sensitivity']['mean']:.2f}% ± "
                  f"{metrics['sensitivity']['std']:.2f}%")
            
            print(f"Specificity: {metrics['specificity']['mean']:.2f}% ± "
                  f"{metrics['specificity']['std']:.2f}%")
            
            print(f"Accuracy: {metrics['accuracy']['mean']:.2f}% ± "
                  f"{metrics['accuracy']['std']:.2f}%")
            
            # Compare with paper results
            print("\nCOMPARISON WITH PAPER RESULTS:")
            print("-" * 40)
            print("Paper reports (average across datasets):")
            print("- Sensitivity: 92.3% (DDSM), 92.11% (MIAS)")
            print("- Specificity: 99.21% (DDSM), 99.45% (MIAS)")
            print("- Accuracy: 98.68% (DDSM), 98.93% (MIAS)")
            
        print("="*60)


def main():
    """Main function for paper-exact training."""
    parser = argparse.ArgumentParser(description='MLMO-EMO Paper Exact Implementation')
    
    parser.add_argument('--csv-path', type=str, required=True,
                       help='Path to dataset CSV file')
    
    parser.add_argument('--method', type=str, default='otsu',
                       choices=['otsu', 'kapur'],
                       help='Thresholding method (otsu or kapur)')
    
    parser.add_argument('--max-images', type=int, default=None,
                       help='Maximum number of images to process')
    
    parser.add_argument('--output-dir', type=str, default='paper_results',
                       help='Output directory for results')
    
    parser.add_argument('--save-detailed', action='store_true',
                       help='Save detailed results for each image')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Verify dataset file exists
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"Dataset CSV not found: {args.csv_path}")
    
    print("Starting MLMO-EMO Paper Exact Implementation")
    print("-" * 50)
    print(f"Dataset CSV: {args.csv_path}")
    print(f"Method: {args.method}")
    print(f"Max images: {args.max_images or 'All'}")
    print(f"Output directory: {args.output_dir}")
    
    # Initialize processor
    processor = PaperDatasetProcessor(args.csv_path)
    
    # Process dataset
    results = processor.process_dataset(
        method=args.method,
        max_images=args.max_images,
        save_results=args.save_detailed
    )
    
    # Print summary
    processor.print_summary(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, 
                              f"mlmo_emo_results_{args.method}_{timestamp}.json")
    processor.save_results(results, output_file)
    
    print(f"\nProcessing complete! Results saved to {output_file}")


if __name__ == "__main__":
    # Example usage for testing without command line args
    if len(os.sys.argv) == 1:
        print("Running in demo mode...")
        
        # Check if dataset CSV exists
        csv_path = "unified_segmentation_dataset.csv"
        if os.path.exists(csv_path):
            processor = PaperDatasetProcessor(csv_path)
            
            # Process a small subset for demo
            results = processor.process_dataset(
                method='otsu', 
                max_images=5,  # Just 5 images for demo
                save_results=True
            )
            
            processor.print_summary(results)
            
            # Save demo results
            os.makedirs('demo_results', exist_ok=True)
            processor.save_results(results, 'demo_results/demo_mlmo_emo_results.json')
            
        else:
            print(f"Dataset CSV not found: {csv_path}")
            print("Please run dataset processing first or use command line arguments.")
            
            # Show usage
            print("\nUsage:")
            print("python paper_exact_training.py --csv-path dataset.csv --method otsu")
    else:
        main()