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
import pickle
import shutil

cv2.setUseOptimized(True)
cv2.setNumThreads(4)  # Use 4 CPU threads

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

# Import adaptive column detection functions
try:
    # Try importing from parent directory
    import sys
    sys.path.append('..')
    from dataset_process import (
        get_image_column_name, 
        get_mask_column_names,
        load_image_adaptive
    )
except ImportError:
    try:
        # Try importing from current directory
        sys.path.append('.')
        from dataset_process import (
            get_image_column_name, 
            get_mask_column_names,
            load_image_adaptive,
            _split_mask_field,
            _resolve_mask_path,
            _merge_mask_files
        )
    except ImportError:
        # Fallback: define the functions locally
        def get_image_column_name(columns, debug=False):
            """Fallback function to detect image column."""
            image_patterns = [
                'image_path', 'image_file_path', 'fullPath', 'fileName', 
                'imagePath', 'file_path', 'path'
            ]
            for pattern in image_patterns:
                for col in columns:
                    if pattern.lower() in col.lower():
                        return col
            return columns[0] if columns else None
            
        def get_mask_column_names(columns, debug=False):
            """Fallback function to detect mask columns."""
            mask_columns = []
            mask_patterns = [
                'mask_path', 'roi_mask_file_path', 'Tumour_Contour', 'mask'
            ]
            for pattern in mask_patterns:
                for col in columns:
                    if pattern.lower() in col.lower():
                        mask_columns.append(col)
            return mask_columns
            
        def load_image_adaptive(path, debug=False):
            """Fallback function to load images."""
            import cv2
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            return img, path if img is not None else (None, None)


class PaperDatasetProcessor:
    """
    Dataset processor that works with the exact paper methodology.
    """
    
    def __init__(self, dataset_csv: str, checkpoint_dir: str = "checkpoints",
                 population_size: int = 30,
                 max_iterations: int = 50,
                 local_search_prob: float = 0.5,
                 local_search_iterations: int = 3):
        """
        Initialize with dataset CSV file.
        
        Args:
            dataset_csv: Path to CSV file with image and mask paths
            checkpoint_dir: Directory to save checkpoints (default: "checkpoints")
            population_size: EML optimizer population size
            max_iterations: EML optimizer max iterations
            local_search_prob: EML local search probability
            local_search_iterations: EML local search iterations
        """
        self.df = pd.read_csv(dataset_csv)
        self.model = PaperSegmentationModel(
            population_size=population_size,
            max_iterations=max_iterations,
            local_search_prob=local_search_prob,
            local_search_iterations=local_search_iterations
        )
        self.metrics_calculator = PaperEvaluationMetrics()
        
        # Setup checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize best model tracking
        self.best_dice = -1
        self.best_model = None
        
        # Detect column names adaptively
        columns = self.df.columns.tolist()
        print(f"Available columns: {columns}")
        
        # Detect image column
        self.image_column = get_image_column_name(columns, debug=True)
        if self.image_column is None:
            raise ValueError(f"Could not detect image column in: {columns}")
        print(f"Using image column: {self.image_column}")
        
        # Detect mask columns
        self.mask_columns = get_mask_column_names(columns, debug=True)
        if not self.mask_columns:
            raise ValueError(f"Could not detect mask columns in: {columns}")
        print(f"Using mask columns: {self.mask_columns}")
        
        # Use the first mask column as primary
        self.mask_column = self.mask_columns[0]
        print(f"Primary mask column: {self.mask_column}")
        
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
            # Load image using adaptive loading
            image, actual_image_path = load_image_adaptive(image_path, debug=False)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")

            # Robust mask loading logic - masks in CSV can be malformed, multiple paths, or relative
            mask = None
            actual_mask_path = None

            # 1) Try direct adaptive load first
            if isinstance(mask_path, str) and mask_path.strip():
                mask, actual_mask_path = load_image_adaptive(mask_path, debug=False)

            # 2) If not found, try splitting multiple mask entries and load/merge
            if mask is None and isinstance(mask_path, str):
                candidates = _split_mask_field(mask_path)
                if candidates:
                    # Try each candidate as absolute/relative path
                    loaded_masks = []
                    for cand in candidates:
                        m, p = load_image_adaptive(cand, debug=False)
                        if m is not None:
                            loaded_masks.append((m, p))
                        else:
                            # Try resolving relative to image directory
                            img_dir = os.path.dirname(actual_image_path or image_path) or '.'
                            resolved = _resolve_mask_path(cand, img_dir)
                            if resolved:
                                m2, p2 = load_image_adaptive(resolved, debug=False)
                                if m2 is not None:
                                    loaded_masks.append((m2, p2))

                    if loaded_masks:
                        # Merge masks into single mask of image.shape
                        mask_list = [p for (_, p) in loaded_masks]
                        # Use _merge_mask_files which expects list of paths
                        mask = _merge_mask_files(mask_list, image.shape, debug=False)
                        actual_mask_path = ','.join(mask_list)

            # 3) If still not found, try resolving single mask path relative to image dir
            if mask is None and isinstance(mask_path, str) and mask_path.strip():
                img_dir = os.path.dirname(actual_image_path or image_path) or '.'
                resolved = _resolve_mask_path(mask_path, img_dir)
                if resolved:
                    mask, actual_mask_path = load_image_adaptive(resolved, debug=False)

            # 4) Final fallback: if mask still None but mask_path is empty/None, create empty mask
            if mask is None:
                # Before failing, provide debug information to help tracing
                print(f"[DEBUG] Failed to load mask for image: {image_path}")
                print(f"[DEBUG] mask_path value: {mask_path}")
                print(f"[DEBUG] image resolved path: {actual_image_path}")
                # Create an empty mask of same shape to allow segmentation and measurement (will yield zero metrics)
                mask = np.zeros_like(image)
                actual_mask_path = None

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
            
            result = {
                'image_path': image_path,
                'mask_path': mask_path,
                'segmentation_results': segmentation_results,
                'template_matching': matching_results,
                'metrics': metrics,
                'success': True,
                'error': None
            }
            
            # Track best model based on dice score
            if result['success'] and result['metrics']:
                dice = result['metrics']['dice_coefficient']
                if dice > self.best_dice:
                    self.best_dice = dice
                    self.best_model = {
                        'thresholds': result['segmentation_results']['thresholds'],
                        'dice': dice,
                        'image_path': result['image_path'],
                        'method': method
                    }
            
            return result
            
        except Exception as e:
            return {
                'image_path': image_path,
                'mask_path': mask_path,
                'success': False,
                'error': str(e),
                'metrics': None
            }
    
    def _save_checkpoint(self, results: List[Dict], all_metrics: List[Dict], 
                        processed_count: int, method: str):
        """
        Save checkpoint to enable resuming from interruptions.
        
        Args:
            results: List of processing results
            all_metrics: List of metrics dictionaries
            processed_count: Number of images processed
            method: Thresholding method used
        """
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{method}_{processed_count}.pkl"
        checkpoint_data = {
            'results': results,
            'all_metrics': all_metrics,
            'processed_count': processed_count,
            'method': method,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        print(f"Checkpoint saved: {checkpoint_file}")
        
        # Keep only last 3 checkpoints to save disk space
        self._cleanup_old_checkpoints(method, keep_last=3)
    
    def _cleanup_old_checkpoints(self, method: str, keep_last: int = 3):
        """Remove old checkpoint files, keeping only the most recent ones."""
        checkpoint_files = sorted(
            self.checkpoint_dir.glob(f"checkpoint_{method}_*.pkl"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        for old_checkpoint in checkpoint_files[keep_last:]:
            try:
                old_checkpoint.unlink()
                print(f"Removed old checkpoint: {old_checkpoint.name}")
            except Exception as e:
                print(f"Warning: Could not remove {old_checkpoint.name}: {e}")
    
    def _load_latest_checkpoint(self, method: str) -> Tuple[List[Dict], List[Dict], int]:
        """
        Load the most recent checkpoint for resuming processing.
        
        Args:
            method: Thresholding method to load checkpoint for
            
        Returns:
            Tuple of (results, all_metrics, processed_count) or ([], [], 0) if no checkpoint
        """
        checkpoint_files = sorted(
            self.checkpoint_dir.glob(f"checkpoint_{method}_*.pkl"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        if not checkpoint_files:
            return [], [], 0
        
        latest_checkpoint = checkpoint_files[0]
        print(f"Loading checkpoint: {latest_checkpoint}")
        
        try:
            with open(latest_checkpoint, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            print(f"Resumed from {checkpoint_data['processed_count']} images")
            print(f"Checkpoint timestamp: {checkpoint_data['timestamp']}")
            
            return (
                checkpoint_data['results'],
                checkpoint_data['all_metrics'],
                checkpoint_data['processed_count']
            )
        except Exception as e:
            print(f"Warning: Could not load checkpoint {latest_checkpoint}: {e}")
            return [], [], 0
    
    def _save_results(self, final_results: Dict, method: str, output_dir: str = "results"):
        """
        Save processing results in multiple formats.
        
        Args:
            final_results: Complete processing results dictionary
            method: Thresholding method used
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"results_{method}_{timestamp}"
        
        # Save as JSON (summary)
        json_file = output_path / f"{base_filename}.json"
        json_data = {
            'method': final_results['method'],
            'total_images': final_results['total_images'],
            'successful_processing': final_results['successful_processing'],
            'success_rate': final_results['success_rate'],
            'aggregate_metrics': final_results['aggregate_metrics'],
            'processing_timestamp': final_results['processing_timestamp']
        }
        
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=4)
        print(f"JSON summary saved: {json_file}")
        
        # Save detailed results as pickle (includes all data)
        pickle_file = output_path / f"{base_filename}_detailed.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(final_results, f)
        print(f"Detailed results saved: {pickle_file}")
        
        # Save metrics as CSV for easy analysis
        if final_results['detailed_results']:
            csv_file = output_path / f"{base_filename}_metrics.csv"
            metrics_rows = []
            
            for result in final_results['detailed_results']:
                if result['success'] and result['metrics']:
                    row = {
                        'image_path': result['image_path'],
                        'mask_path': result['mask_path'],
                        **result['metrics']
                    }
                    metrics_rows.append(row)
            
            if metrics_rows:
                metrics_df = pd.DataFrame(metrics_rows)
                metrics_df.to_csv(csv_file, index=False)
                print(f"Metrics CSV saved: {csv_file}")
        
        # Save best model if found
        if self.best_model:
            best_file = output_path / f"best_model_{method}_{timestamp}.pkl"
            with open(best_file, 'wb') as f:
                pickle.dump(self.best_model, f)
            print(f"Best model saved: {best_file}")
        
        return json_file, pickle_file
    
    def process_dataset(self, method: str = 'otsu', 
                       max_images: int = None,
                       save_results: bool = True,
                       resume_from_checkpoint: bool = True,
                       checkpoint_interval: int = 10) -> Dict:
        """
        Process entire dataset using paper methodology with checkpointing.
        
        Args:
            method: 'otsu' or 'kapur' thresholding method
            max_images: Maximum number of images to process (None for all)
            save_results: Whether to save detailed results
            resume_from_checkpoint: Whether to resume from last checkpoint
            checkpoint_interval: Save checkpoint every N images (default: 10)
            
        Returns:
            Aggregated results and statistics
        """
        print(f"Processing dataset with {method} method...")
        print(f"Total images in dataset: {len(self.df)}")
        
        # Limit number of images if specified
        df_subset = self.df.head(max_images) if max_images else self.df
        
        # Try to load checkpoint if resuming
        results = []
        all_metrics = []
        start_idx = 0
        
        if resume_from_checkpoint:
            results, all_metrics, start_idx = self._load_latest_checkpoint(method)
            if start_idx > 0:
                print(f"Resuming from image {start_idx + 1}/{len(df_subset)}")
        
        # Process each image with progress bar
        for idx, row in tqdm(df_subset.iterrows(), 
                           total=len(df_subset),
                           desc="Processing images",
                           initial=start_idx):
            
            # Skip already processed images
            if idx < start_idx:
                continue
            
            # Use adaptive column names
            image_path = row[self.image_column]
            mask_path = row[self.mask_column]
            
            # Process single image
            result = self.process_single_image(image_path, mask_path, method)
            results.append(result)
            
            # Collect metrics for successful processing
            if result['success'] and result['metrics']:
                all_metrics.append(result['metrics'])
            
            # Save checkpoint at regular intervals
            if (idx + 1) % checkpoint_interval == 0:
                success_rate = sum(1 for r in results if r['success']) / len(results)
                print(f"Processed {idx + 1} images, success rate: {success_rate:.2%}")
                self._save_checkpoint(results, all_metrics, idx + 1, method)
        
        # Save final checkpoint
        self._save_checkpoint(results, all_metrics, len(df_subset), method)
        
        # Calculate aggregate statistics
        aggregate_stats = self._calculate_aggregate_statistics(all_metrics)
        
        # Prepare final results
        final_results = {
            'method': method,
            'total_images': len(df_subset),
            'successful_processing': len(all_metrics),
            'success_rate': len(all_metrics) / len(df_subset) if len(df_subset) > 0 else 0,
            'aggregate_metrics': aggregate_stats,
            'detailed_results': results if save_results else None,
            'processing_timestamp': datetime.now().isoformat()
        }
        
        # Save results in multiple formats
        if save_results:
            self._save_results(final_results, method)
            print(f"\n{'='*60}")
            print("Results saved successfully!")
            print(f"{'='*60}")
        
        return final_results

    def quick_validate(self, n: int = 5, method: str = 'otsu') -> List[Dict]:
        """
        Quick validation helper to process first `n` images with extra debug prints.
        Returns list of per-image results.
        """
        df_subset = self.df.head(n)
        results = []
        for idx, row in df_subset.iterrows():
            image_path = row[self.image_column]
            mask_path = row[self.mask_column]
            print(f"\n[VALIDATE] Processing {idx+1}: image={image_path}, mask={mask_path}")
            res = self.process_single_image(image_path, mask_path, method)
            if not res['success']:
                print(f"[VALIDATE] ERROR for {image_path}: {res.get('error')}")
            else:
                try:
                    print(f"[VALIDATE] Metrics: {res['metrics']}")
                except Exception:
                    pass
            results.append(res)
        return results
    
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
    
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory for saving checkpoints (default: checkpoints)')
    
    parser.add_argument('--checkpoint-interval', type=int, default=10,
                       help='Save checkpoint every N images (default: 10)')
    
    parser.add_argument('--population-size', type=int, default=30,
                       help='EML optimizer population size (default: 30, reduced for speed)')
    
    parser.add_argument('--max-iterations', type=int, default=50,
                       help='EML optimizer max iterations (default: 50, reduced for speed)')
    
    parser.add_argument('--local-search-prob', type=float, default=0.5,
                       help='EML local search probability (default: 0.5)')
    
    parser.add_argument('--local-search-iterations', type=int, default=3,
                       help='EML local search iterations (default: 3)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Verify dataset file exists
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"Dataset CSV not found: {args.csv_path}")
    
    print("Starting MLMO-EMO Paper Exact Implementation")
    print("=" * 60)
    print(f"Dataset CSV: {args.csv_path}")
    print(f"Method: {args.method}")
    print(f"Max images: {args.max_images or 'All'}")
    print(f"Output directory: {args.output_dir}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Checkpoint interval: {args.checkpoint_interval} images")
    print(f"Resume from checkpoint: {not args.no_resume}")
    print("=" * 60)
    
    # Initialize processor with checkpoint directory
    processor = PaperDatasetProcessor(args.csv_path, 
                                     checkpoint_dir=args.checkpoint_dir,
                                     population_size=args.population_size,
                                     max_iterations=args.max_iterations,
                                     local_search_prob=args.local_search_prob,
                                     local_search_iterations=args.local_search_iterations)
    
    # Process dataset with checkpointing
    results = processor.process_dataset(
        method=args.method,
        max_images=args.max_images,
        save_results=args.save_detailed,
        resume_from_checkpoint=not args.no_resume,
        checkpoint_interval=args.checkpoint_interval
    )
    
    # Print summary
    processor.print_summary(results)
    
    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"{'='*60}")


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