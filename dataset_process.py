#!/usr/bin/env python3
"""
Unified dataset prep for CBIS-DDSM and Mini-DDSM.

This version implements comprehensive image preprocessing based on research paper:
 - Linear and non-linear (sigmoid) normalization techniques
 - Median filtering for noise reduction
 - CLAHE (Contrast Limited Adaptive Histogram Equalization) with proper clip limit calculation
 - Processes CBIS-DDSM and Mini-DDSM datasets
"""
from __future__ import annotations
import os
import argparse
import pandas as pd
import numpy as np
import cv2
from typing import Tuple

# ---------------- Utilities ---------------- #
def ensure_dir(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# ---------------- Image Denoising & Enhancement Techniques ---------------- #

def linear_normalization(image: np.ndarray, new_min: float = 0.0, new_max: float = 255.0) -> np.ndarray:
    """
    Linear normalization technique as per Equation (1):
    Normalization = (I - Min) × (newMax - newMin) / (Max - Min) + newMin
    
    Args:
        image: Input mammographic image
        new_min: Target minimum pixel value (default: 0)
        new_max: Target maximum pixel value (default: 255)
    
    Returns:
        Normalized image
    """
    img_min = np.min(image)
    img_max = np.max(image)
    
    if img_max == img_min:
        # Avoid division by zero
        return np.full_like(image, new_min, dtype=np.float32)
    
    # Apply linear normalization formula
    normalized = (image - img_min) * (new_max - new_min) / (img_max - img_min) + new_min
    
    return normalized.astype(np.float32)


def sigmoid_normalization(image: np.ndarray, alpha: float = 50.0, beta: float = None, 
                         new_min: float = 0.0, new_max: float = 255.0) -> np.ndarray:
    """
    Non-linear (sigmoid) normalization technique as per Equation (2):
    Normalization = (newMax - newMin) × 1 / (1 + e^(-(I - β)/α)) + newMin
    
    Args:
        image: Input mammographic image
        alpha: Width of pixel value (default: 50.0)
        beta: Centered pixel value (default: mean of image)
        new_min: Target minimum pixel value (default: 0)
        new_max: Target maximum pixel value (default: 255)
    
    Returns:
        Sigmoid normalized image
    """
    if beta is None:
        beta = np.mean(image)
    
    # Apply sigmoid normalization formula
    sigmoid_component = 1.0 / (1.0 + np.exp(-(image - beta) / alpha))
    normalized = (new_max - new_min) * sigmoid_component + new_min
    
    return normalized.astype(np.float32)


def median_filtering(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Median filtering technique for noise reduction as per Equation (3).
    
    The median filter is an order statistics filter that replaces the 
    neighbourhood pixel value by the median pixel intensity value.
    
    Note: median[A(x) + B(x)] ≠ median[A(x)] + median[B(x)]
    
    Args:
        image: Input mammographic image
        kernel_size: Size of the median filter kernel (default: 5)
    
    Returns:
        Filtered image
    """
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure odd kernel size
    
    filtered = cv2.medianBlur(image.astype(np.uint8), kernel_size)
    return filtered


def calculate_clahe_clip_limit(M: int, N: int, L: int = 256, delta: float = 0.01) -> float:
    """
    Calculate CLAHE clip limit as per Equation (4):
    
    nT = {
        1,              if δ×M×N/L < 1
        δ×M×N/L,        else, where 0 < δ ≤ 1
    }
    
    Args:
        M: Number of rows in contextual region (tile height)
        N: Number of columns in contextual region (tile width)
        L: Number of histogram bins (default: 256)
        delta: User-defined contrast factor, 0 < δ ≤ 1 (default: 0.01)
    
    Returns:
        Clip limit nT
    """
    clip_value = (delta * M * N) / L
    
    if clip_value < 1:
        n_T = 1.0
    else:
        n_T = clip_value
    
    return n_T


def apply_clahe(image: np.ndarray, tile_grid_size: Tuple[int, int] = (8, 8), 
                delta: float = 0.01) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) with 
    proper clip limit calculation as per Equations (4)-(8).
    
    CLAHE divides the image into non-overlapping blocks (tiles/contextual regions),
    calculates histogram values, and clips them based on user-defined clip limit.
    
    Process:
    1. Calculate clip limit using Equation (4)
    2. Clip original histogram using Equation (5)
    3. Calculate clipped pixels using Equation (6)
    4. Redistribute clipped pixels using Equation (7)
    5. Apply renormalized histogram using Equation (8)
    
    Args:
        image: Input mammographic image (grayscale)
        tile_grid_size: Size of grid for histogram equalization (M, N)
        delta: Contrast factor, 0 < δ ≤ 1 (default: 0.01)
    
    Returns:
        CLAHE enhanced image
    """
    # Ensure image is uint8
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    
    # Get tile dimensions
    img_height, img_width = image.shape
    tile_rows, tile_cols = tile_grid_size
    
    # Calculate contextual region dimensions
    M = img_height // tile_rows  # Rows in each tile
    N = img_width // tile_cols    # Columns in each tile
    L = 256  # Number of histogram bins for 8-bit image
    
    # Calculate clip limit using Equation (4)
    clip_limit = calculate_clahe_clip_limit(M, N, L, delta)
    
    # Create CLAHE object with calculated clip limit
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    # Apply CLAHE
    enhanced = clahe.apply(image)
    
    return enhanced


def preprocess_image(img: np.ndarray, 
                    use_sigmoid: bool = False,
                    median_kernel: int = 5,
                    clahe_tile_size: Tuple[int, int] = (8, 8),
                    clahe_delta: float = 0.01) -> np.ndarray:
    """
    Comprehensive preprocessing pipeline for mammographic images:
    
    1. Normalization (Linear or Sigmoid)
    2. Median Filtering (Noise Reduction)
    3. CLAHE (Contrast Enhancement)
    
    Args:
        img: Input mammographic image
        use_sigmoid: Use sigmoid normalization instead of linear (default: False)
        median_kernel: Kernel size for median filtering (default: 5)
        clahe_tile_size: Tile grid size for CLAHE (default: (8, 8))
        clahe_delta: Contrast factor for CLAHE, 0 < δ ≤ 1 (default: 0.01)
    
    Returns:
        Preprocessed uint8 grayscale image
    """
    # Step 1: Normalization
    if use_sigmoid:
        # Non-linear sigmoid normalization (Equation 2)
        normalized = sigmoid_normalization(img, alpha=50.0, new_min=0.0, new_max=255.0)
    else:
        # Linear normalization (Equation 1)
        normalized = linear_normalization(img, new_min=0.0, new_max=255.0)
    
    # Convert to uint8 for subsequent operations
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    
    # Step 2: Median Filtering (Equation 3)
    # Removes noise from the images using order statistics
    filtered = median_filtering(normalized, kernel_size=median_kernel)
    
    # Step 3: CLAHE (Equations 4-8)
    # Enhances brightness and contrast for better segmentation
    enhanced = apply_clahe(filtered, tile_grid_size=clahe_tile_size, delta=clahe_delta)
    
    return enhanced.astype(np.uint8)

# ---------------- CBIS-DDSM processing ---------------- #
def process_cbis(input_csv, mask_outdir, image_outdir, preproc_args: dict):
    """Process CBIS-DDSM dataset with comprehensive preprocessing."""
    df = pd.read_csv(input_csv)
    ensure_dir(mask_outdir)
    ensure_dir(image_outdir)
    rows = []
    
    print(f"[INFO] Preprocessing settings:")
    print(f"  - Normalization: {'Sigmoid (non-linear)' if preproc_args['use_sigmoid'] else 'Linear'}")
    print(f"  - Median kernel: {preproc_args['median_kernel']}")
    print(f"  - CLAHE tile size: {preproc_args['clahe_tile_size']}")
    print(f"  - CLAHE delta (δ): {preproc_args['clahe_delta']}")
    
    grouped = df.groupby(["patient_id", "image_file_path"], dropna=False)
    for (pid, img_path), group in grouped:
        base_row = group.iloc[0].to_dict()
        abnormality_ids = group["abnormality_id"].astype(str).unique().tolist()
        mask_paths = [mp for mp in group["roi_mask_file_path"].dropna().unique().tolist() if isinstance(mp, str)]
        
        # Merge masks if multiple
        merged_mask = None
        for mp in mask_paths:
            if not os.path.exists(mp):
                continue
            mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            mask = (mask > 0).astype(np.uint8) * 255
            merged_mask = mask if merged_mask is None else cv2.bitwise_or(merged_mask, mask)
        
        # Load and preprocess image
        if os.path.exists(img_path):
            full_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if full_img is None:
                continue
            # Apply comprehensive preprocessing pipeline
            processed_img = preprocess_image(
                full_img,
                use_sigmoid=preproc_args['use_sigmoid'],
                median_kernel=preproc_args['median_kernel'],
                clahe_tile_size=tuple(preproc_args['clahe_tile_size']),
                clahe_delta=preproc_args['clahe_delta']
            )
        else:
            continue
        
        if merged_mask is None:
            merged_mask = np.zeros_like(processed_img, dtype=np.uint8)
        
        # Ensure shapes match
        if merged_mask.shape != processed_img.shape:
            merged_mask = cv2.resize(merged_mask, (processed_img.shape[1], processed_img.shape[0]), 
                                    interpolation=cv2.INTER_NEAREST)
        
        # Save with unique name
        basename = os.path.splitext(os.path.basename(img_path))[0]
        abn_str = "-".join(abnormality_ids) if abnormality_ids else "NA"
        unique_name = f"CBIS_{pid}_{basename}_{abn_str}"
        
        proc_img_path = os.path.join(image_outdir, f"{unique_name}.png")
        mask_path = os.path.join(mask_outdir, f"{unique_name}_mask.png")
        
        cv2.imwrite(proc_img_path, processed_img)
        cv2.imwrite(mask_path, merged_mask)
        
        base_row["dataset"] = "CBIS-DDSM"
        base_row["image_file_path"] = proc_img_path
        base_row["roi_mask_file_path"] = mask_path
        rows.append(base_row)
    
    return pd.DataFrame(rows)

# ---------------- Mini-DDSM processing ---------------- #
def process_mini_ddsm(excel_path, base_dir, mask_outdir, image_outdir, preproc_args: dict):
    """
    Process Mini-DDSM dataset with comprehensive preprocessing.
    
    Args:
        excel_path: Path to DataWMask.xlsx file
        base_dir: Base directory containing the Mini-DDSM images
        mask_outdir: Output directory for processed masks
        image_outdir: Output directory for processed images
        preproc_args: Dictionary containing preprocessing parameters
    """
    ensure_dir(mask_outdir)
    ensure_dir(image_outdir)
    
    # Read the Data sheet from Excel
    df = pd.read_excel(excel_path, sheet_name="Data")
    rows = []
    
    for idx, row in df.iterrows():
        img_rel_path = row["fullPath"]
        img_path = os.path.join(base_dir, img_rel_path)
        
        if not os.path.exists(img_path):
            print(f"[WARNING] Image not found: {img_path}")
            continue
        
        # Load and preprocess image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARNING] Failed to load image: {img_path}")
            continue
        
        # Apply comprehensive preprocessing pipeline
        processed_img = preprocess_image(
            img,
            use_sigmoid=preproc_args['use_sigmoid'],
            median_kernel=preproc_args['median_kernel'],
            clahe_tile_size=tuple(preproc_args['clahe_tile_size']),
            clahe_delta=preproc_args['clahe_delta']
        )
        
        # Load mask if available
        mask = np.zeros_like(processed_img, dtype=np.uint8)
        tumour_contour = row.get("Tumour_Contour", None)
        tumour_contour2 = row.get("Tumour_Contour2", None)
        
        # Check if mask paths are available (not NaN and not "-")
        if pd.notna(tumour_contour) and str(tumour_contour) != "-":
            mask_path = os.path.join(base_dir, tumour_contour)
            if os.path.exists(mask_path):
                mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask_img is not None:
                    mask_img = (mask_img > 0).astype(np.uint8) * 255
                    if mask_img.shape != processed_img.shape:
                        mask_img = cv2.resize(mask_img, (processed_img.shape[1], processed_img.shape[0]), 
                                            interpolation=cv2.INTER_NEAREST)
                    mask = cv2.bitwise_or(mask, mask_img)
        
        # Check for second mask if available
        if pd.notna(tumour_contour2) and str(tumour_contour2) != "-":
            mask_path2 = os.path.join(base_dir, tumour_contour2)
            if os.path.exists(mask_path2):
                mask_img2 = cv2.imread(mask_path2, cv2.IMREAD_GRAYSCALE)
                if mask_img2 is not None:
                    mask_img2 = (mask_img2 > 0).astype(np.uint8) * 255
                    if mask_img2.shape != processed_img.shape:
                        mask_img2 = cv2.resize(mask_img2, (processed_img.shape[1], processed_img.shape[0]), 
                                             interpolation=cv2.INTER_NEAREST)
                    mask = cv2.bitwise_or(mask, mask_img2)
        
        # Create unique filename
        filename = row["fileName"]
        basename = os.path.splitext(filename)[0]
        status = row["Status"]
        side = row["Side"]
        view = row["View"]
        unique_name = f"MINI_{status}_{basename}"
        
        proc_img_path = os.path.join(image_outdir, f"{unique_name}.png")
        mask_path_out = os.path.join(mask_outdir, f"{unique_name}_mask.png")
        
        cv2.imwrite(proc_img_path, processed_img)
        cv2.imwrite(mask_path_out, mask)
        
        # Create row data
        row_data = {
            "dataset": "Mini-DDSM",
            "patient_id": unique_name,
            "image_file_path": proc_img_path,
            "roi_mask_file_path": mask_path_out,
            "pathology": status,
            "abnormality_id": status,
            "side": side,
            "view": view,
            "age": row.get("Age", None),
            "density": row.get("Density", None),
        }
        rows.append(row_data)
    
    return pd.DataFrame(rows)

# ---------------- Main Processing Function ---------------- #
def process_datasets(cbis_csv, mini_ddsm_excel, mini_ddsm_base_dir, output_csv, outdir, preproc_args: dict):
    """
    Process CBIS-DDSM and Mini-DDSM datasets with comprehensive preprocessing.
    
    Args:
        cbis_csv: Path to CBIS-DDSM CSV file
        mini_ddsm_excel: Path to Mini-DDSM DataWMask.xlsx file
        mini_ddsm_base_dir: Base directory containing Mini-DDSM images
        output_csv: Output path for unified CSV
        outdir: Output directory for processed images and masks
        preproc_args: Dictionary containing preprocessing parameters
    """
    cbis_img_dir = os.path.join(outdir, "CBIS_IMAGES")
    cbis_mask_dir = os.path.join(outdir, "CBIS_MASKS")
    mini_img_dir = os.path.join(outdir, "MINI_IMAGES")
    mini_mask_dir = os.path.join(outdir, "MINI_MASKS")

    ensure_dir(cbis_img_dir)
    ensure_dir(cbis_mask_dir)
    ensure_dir(mini_img_dir)
    ensure_dir(mini_mask_dir)

    print("[INFO] Processing CBIS-DDSM dataset...")
    cbis_df = process_cbis(cbis_csv, cbis_mask_dir, cbis_img_dir, preproc_args)
    
    print("[INFO] Processing Mini-DDSM dataset...")
    mini_df = process_mini_ddsm(mini_ddsm_excel, mini_ddsm_base_dir, mini_mask_dir, mini_img_dir, preproc_args)

    # Merge datasets
    merged = pd.concat([cbis_df, mini_df], ignore_index=True)
    merged.to_csv(output_csv, index=False)
    
    print(f"\n[INFO] Unified dataset saved → {output_csv}")
    print(f"[INFO] Total samples: {len(merged)} "
          f"(CBIS-DDSM={len(cbis_df)}, Mini-DDSM={len(mini_df)})")
    print(f"\n[INFO] Preprocessing configuration:")
    print(f"  - Normalization: {'Sigmoid (non-linear)' if preproc_args['use_sigmoid'] else 'Linear'}")
    print(f"  - Median kernel: {preproc_args['median_kernel']}")
    print(f"  - CLAHE tile size: {preproc_args['clahe_tile_size']}")
    print(f"  - CLAHE delta (δ): {preproc_args['clahe_delta']}")

# ---------------- CLI ---------------- #
def get_args():
    p = argparse.ArgumentParser(
        description="Prepare CBIS-DDSM + Mini-DDSM unified dataset with comprehensive preprocessing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Preprocessing Pipeline:
  1. Normalization (Linear or Sigmoid)
     - Linear: Normalization = (I - Min) × (newMax - newMin) / (Max - Min) + newMin
     - Sigmoid: Normalization = (newMax - newMin) × 1/(1 + e^(-(I-β)/α)) + newMin
  
  2. Median Filtering
     - Order statistics filter for noise reduction
     - Replaces neighbourhood pixels with median intensity
  
  3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
     - Clip limit: nT = δ×M×N/L (with constraints)
     - Enhances brightness and contrast for better segmentation

Examples:
  # Basic usage with default preprocessing
  python dataset_process.py --cbis-csv data.csv --mini-ddsm-excel DataWMask.xlsx \\
    --mini-ddsm-base-dir ./mini-ddsm --output-csv unified.csv
  
  # Use sigmoid normalization with custom parameters
  python dataset_process.py --cbis-csv data.csv --mini-ddsm-excel DataWMask.xlsx \\
    --mini-ddsm-base-dir ./mini-ddsm --output-csv unified.csv \\
    --use-sigmoid --median-kernel 7 --clahe-delta 0.02
        """)
    
    # Required arguments
    p.add_argument("--cbis-csv", type=str, required=True, 
                   help="Path to CBIS-DDSM CSV file")
    p.add_argument("--mini-ddsm-excel", type=str, required=True,
                   help="Path to Mini-DDSM DataWMask.xlsx file")
    p.add_argument("--mini-ddsm-base-dir", type=str, required=True,
                   help="Base directory containing Mini-DDSM images")
    p.add_argument("--output-csv", type=str, required=True,
                   help="Output path for unified CSV")
    p.add_argument("--outdir", type=str, default="DATASET",
                   help="Output directory for processed images and masks (default: DATASET)")
    
    # Preprocessing arguments
    preproc = p.add_argument_group('preprocessing options')
    preproc.add_argument("--use-sigmoid", action="store_true",
                        help="Use sigmoid (non-linear) normalization instead of linear")
    preproc.add_argument("--median-kernel", type=int, default=5,
                        help="Kernel size for median filtering (must be odd, default: 5)")
    preproc.add_argument("--clahe-tile-size", type=int, nargs=2, default=[8, 8],
                        metavar=("ROWS", "COLS"),
                        help="CLAHE tile grid size (default: 8 8)")
    preproc.add_argument("--clahe-delta", type=float, default=0.01,
                        help="CLAHE contrast factor δ, where 0 < δ ≤ 1 (default: 0.01)")
    
    return p.parse_args()

if __name__ == "__main__":
    args = get_args()
    
    # Validate preprocessing arguments
    if args.clahe_delta <= 0 or args.clahe_delta > 1:
        raise ValueError(f"CLAHE delta (δ) must be in range (0, 1], got {args.clahe_delta}")
    
    if args.median_kernel % 2 == 0:
        print(f"[WARNING] Median kernel size must be odd, adjusting {args.median_kernel} to {args.median_kernel + 1}")
        args.median_kernel += 1
    
    # Prepare preprocessing arguments dictionary
    preproc_args = {
        'use_sigmoid': args.use_sigmoid,
        'median_kernel': args.median_kernel,
        'clahe_tile_size': args.clahe_tile_size,
        'clahe_delta': args.clahe_delta
    }
    
    # Process datasets
    process_datasets(
        args.cbis_csv,
        args.mini_ddsm_excel,
        args.mini_ddsm_base_dir,
        args.output_csv,
        args.outdir,
        preproc_args
    )
