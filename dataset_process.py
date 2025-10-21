#!/usr/bin/env python3
"""
Unified dataset prep for CBIS-DDSM, Mini-DDSM, and Mini-DDSM Data-MoreThanTwoMasks.

This version implements comprehensive image preprocessing based on research paper:
 - Linear and non-linear (sigmoid) normalization techniques
 - Median filtering for noise reduction
 - CLAHE (Contrast Limited Adaptive Histogram Equalization) with proper clip limit calculation
 - Processes CBIS-DDSM, Mini-DDSM (DataWMask.xlsx), and Mini-DDSM Data-MoreThanTwoMasks datasets
 - Supports multiple mask contours (Tumour_Contour, Tumour_Contour2, Tumour_Contour3)
 - Merges multiple masks per image into a single binary mask

Behavior:
 - Comprehensive preprocessing with research-backed techniques
 - Merges multiple masks per image into a single binary mask
 - Supports MINI and MINI_MORE (Data-MoreThanTwoMasks) in different base directories
 - Produces processed images and masks into output subdirectories and writes a unified CSV

Usage example:
 python dataset_process.py \
   --cbis-csv /path/to/cbis.csv \
   --mini-ddsm-excel /path/to/DataWMask.xlsx \
   --mini-ddsm-base-dir /path/to/MINI_IMAGES_BASE \
   --mini2-excel /path/to/Data-MoreThanTwoMasks.xlsx \
   --mini2-base-dir /path/to/MINI2_BASE \
   --outdir OUTDIR \
   --output-csv unified.csv --use-sigmoid
"""
from __future__ import annotations
import os
import argparse
import pandas as pd
import numpy as np
import cv2
from typing import Tuple, List, Optional

# ---------------- Utilities ---------------- #
def ensure_dir(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def _normalize_mask_ref(mask_ref: str) -> Optional[str]:
    """Return None for empty/placeholder tokens, else normalized path string."""
    if mask_ref is None:
        return None
    s = str(mask_ref).strip()
    if not s or s in ("-", "nan", "None", "NULL"):
        return None
    return s

def _split_mask_field(s: str) -> List[str]:
    """
    A mask field may contain multiple mask paths separated by
    ';', '|', ',', or whitespace. Return list of cleaned strings.
    """
    if s is None:
        return []
    parts = []
    for token in [p for sep in [';', '|', ','] for p in s.split(sep)]:
        token = token.strip()
        if token:
            parts.append(token)
    # If above split produced nothing but original has spaces, try whitespace split
    if not parts and isinstance(s, str):
        parts = [p for p in s.split() if p]
    return parts

def _resolve_mask_path(candidate: str, base_dir: str) -> Optional[str]:
    """
    Try to resolve a mask candidate path to an existing file.
    Handles backslashes, leading/trailing slashes, and absolute paths.
    """
    if candidate is None:
        return None
    cand = candidate.strip()
    if not cand or cand in ("-", "nan", "None", "NULL"):
        return None

    # If absolute path and exists
    if os.path.isabs(cand) and os.path.exists(cand):
        return cand

    # Try as relative to base_dir (preserve internal backslashes)
    joined = os.path.join(base_dir, cand)
    norm = os.path.normpath(joined)
    if os.path.exists(norm):
        return norm

    # Try replace backslashes with os.sep pieces
    cand2 = cand.replace("\\", os.sep).replace("/", os.sep)
    joined2 = os.path.join(base_dir, cand2)
    norm2 = os.path.normpath(joined2)
    if os.path.exists(norm2):
        return norm2

    # Try only the filename inside base_dir subtree (slow for large bases; we avoid scanning entire tree)
    # but attempt a sibling check: if candidate looks like "subdir/file.png", try last two components
    parts = cand.replace("\\", "/").split("/")
    if len(parts) >= 2:
        tail = os.path.join(*parts[-2:])
        cand3 = os.path.join(base_dir, tail)
        if os.path.exists(cand3):
            return os.path.normpath(cand3)

    # Not found
    return None

def _merge_mask_files(mask_paths: List[str], target_shape: tuple) -> np.ndarray:
    """Read and OR all existing masks in mask_paths; return binary (0/255) mask shaped target_shape."""
    h, w = target_shape
    out_mask = np.zeros((h, w), dtype=np.uint8)
    for mp in mask_paths:
        if not mp:
            continue
        if not os.path.exists(mp):
            continue
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        # Binarize then resize if needed
        m_bin = (m > 0).astype(np.uint8) * 255
        if m_bin.shape != (h, w):
            m_bin = cv2.resize(m_bin, (w, h), interpolation=cv2.INTER_NEAREST)
        out_mask = cv2.bitwise_or(out_mask, m_bin)
    return out_mask

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
def process_mini_ddsm(excel_path, base_dir, mask_outdir, image_outdir, preproc_args: dict,
                      contour_columns: Optional[List[str]] = None):
    """
    Process Mini-DDSM dataset with comprehensive preprocessing.
    Supports multiple mask columns (Tumour_Contour, Tumour_Contour2, Tumour_Contour3, etc.)
    
    Args:
        excel_path: Path to DataWMask.xlsx or Data-MoreThanTwoMasks.xlsx file
        base_dir: Base directory containing the Mini-DDSM images
        mask_outdir: Output directory for processed masks
        image_outdir: Output directory for processed images
        preproc_args: Dictionary containing preprocessing parameters
        contour_columns: List of column names to look for masks (e.g. ["Tumour_Contour","Tumour_Contour2"])
                        If None, defaults to ["Tumour_Contour","Tumour_Contour2"]
    """
    if contour_columns is None:
        contour_columns = ["Tumour_Contour", "Tumour_Contour2"]
    
    ensure_dir(mask_outdir)
    ensure_dir(image_outdir)
    
    # Read the Data sheet from Excel
    df = pd.read_excel(excel_path, sheet_name="Data") if excel_path.endswith((".xlsx", ".xls")) else pd.read_csv(excel_path)
    rows = []
    
    for idx, row in df.iterrows():
        img_rel_path = row.get("fullPath") or row.get("fileName")  # prefer fullPath then fileName
        if pd.isna(img_rel_path):
            continue
        img_rel_path = str(img_rel_path).strip()
        img_path = os.path.join(base_dir, img_rel_path) if not os.path.isabs(img_rel_path) else img_rel_path
        img_path = os.path.normpath(img_path)
        
        if not os.path.exists(img_path):
            print(f"[WARNING] MINI: Image not found: {img_path}")
            continue
        
        # Load and preprocess image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARNING] MINI: Failed to load image: {img_path}")
            continue
        
        # Apply comprehensive preprocessing pipeline
        processed_img = preprocess_image(
            img,
            use_sigmoid=preproc_args['use_sigmoid'],
            median_kernel=preproc_args['median_kernel'],
            clahe_tile_size=tuple(preproc_args['clahe_tile_size']),
            clahe_delta=preproc_args['clahe_delta']
        )
        
        h, w = processed_img.shape[:2]
        
        # Collect mask candidate strings across provided contour columns
        mask_candidates = []
        for col in contour_columns:
            if col in row:
                raw = row.get(col)
                raw = _normalize_mask_ref(raw)
                if raw:
                    parts = _split_mask_field(str(raw))
                    mask_candidates.extend(parts)
        
        # Resolve candidate paths into actual existing files
        resolved_masks = []
        for cand in mask_candidates:
            resolved = _resolve_mask_path(cand, base_dir)
            if resolved:
                resolved_masks.append(resolved)
            else:
                # Try if cand is just filename in same folder as image
                possible = os.path.join(os.path.dirname(img_path), cand)
                if os.path.exists(possible):
                    resolved_masks.append(os.path.normpath(possible))
                else:
                    # Not found: continue without warning for cleaner output
                    pass
        
        # Merge masks (if any) using the utility function
        mask = _merge_mask_files(resolved_masks, (h, w))
        
        # Create unique filename
        filename = row.get("fileName") or os.path.basename(img_path)
        basename = os.path.splitext(filename)[0]
        status = row.get("Status", "")
        side = row.get("Side", "")
        view = row.get("View", "")
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
def process_datasets(cbis_csv, mini_ddsm_excel, mini_ddsm_base_dir, 
                     mini2_excel, mini2_base_dir, output_csv, outdir, preproc_args: dict):
    """
    Process CBIS-DDSM, Mini-DDSM, and optionally Data-MoreThanTwoMasks (mini2) datasets 
    with comprehensive preprocessing.
    
    Args:
        cbis_csv: Path to CBIS-DDSM CSV file
        mini_ddsm_excel: Path to Mini-DDSM DataWMask.xlsx file
        mini_ddsm_base_dir: Base directory containing Mini-DDSM images
        mini2_excel: Path to Data-MoreThanTwoMasks.xlsx file (optional)
        mini2_base_dir: Base directory for Data-MoreThanTwoMasks images/masks (optional)
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
    
    print("[INFO] Processing Mini-DDSM (DataWMask.xlsx) dataset...")
    mini_df = process_mini_ddsm(mini_ddsm_excel, mini_ddsm_base_dir, mini_mask_dir, mini_img_dir, preproc_args,
                                contour_columns=["Tumour_Contour", "Tumour_Contour2"])

    mini2_df = pd.DataFrame([])
    if mini2_excel and mini2_base_dir:
        mini2_img_dir = os.path.join(outdir, "MINI2_IMAGES")
        mini2_mask_dir = os.path.join(outdir, "MINI2_MASKS")
        ensure_dir(mini2_img_dir)
        ensure_dir(mini2_mask_dir)
        print("[INFO] Processing Mini-DDSM Data-MoreThanTwoMasks (supports 3+ masks)...")
        # Pass in third contour column as well
        mini2_df = process_mini_ddsm(mini2_excel, mini2_base_dir, mini2_mask_dir, mini2_img_dir, preproc_args,
                                     contour_columns=["Tumour_Contour", "Tumour_Contour2", "Tumour_Contour3"])
        # Relabel dataset name to distinguish
        if not mini2_df.empty:
            mini2_df["dataset"] = "Mini-DDSM-MoreThanTwoMasks"

    # Concatenate all datasets
    dfs = [df for df in (cbis_df, mini_df, mini2_df) if df is not None and not df.empty]
    if dfs:
        merged = pd.concat(dfs, ignore_index=True)
    else:
        merged = pd.DataFrame([])

    # Save CSV
    ensure_dir(os.path.dirname(os.path.abspath(output_csv)) or ".")
    merged.to_csv(output_csv, index=False)
    
    print(f"\n[INFO] Unified dataset saved → {output_csv}")
    print(f"[INFO] Total samples: {len(merged)} "
          f"(CBIS-DDSM={len(cbis_df)}, Mini-DDSM={len(mini_df)}, Mini-DDSM-MoreThanTwoMasks={len(mini2_df)})")
    print(f"\n[INFO] Preprocessing configuration:")
    print(f"  - Normalization: {'Sigmoid (non-linear)' if preproc_args['use_sigmoid'] else 'Linear'}")
    print(f"  - Median kernel: {preproc_args['median_kernel']}")
    print(f"  - CLAHE tile size: {preproc_args['clahe_tile_size']}")
    print(f"  - CLAHE delta (δ): {preproc_args['clahe_delta']}")

# ---------------- CLI ---------------- #
def get_args():
    p = argparse.ArgumentParser(
        description="Prepare CBIS-DDSM + Mini-DDSM unified dataset with comprehensive preprocessing (supports Data-MoreThanTwoMasks)",
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
  
  # Include Data-MoreThanTwoMasks dataset
  python dataset_process.py --cbis-csv data.csv --mini-ddsm-excel DataWMask.xlsx \\
    --mini-ddsm-base-dir ./mini-ddsm --mini2-excel Data-MoreThanTwoMasks.xlsx \\
    --mini2-base-dir ./mini2-ddsm --output-csv unified.csv
  
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
                   help="Base directory containing Mini-DDSM images (MINI JPEGs)")
    p.add_argument("--mini2-excel", type=str, required=False, default="",
                   help="Path to Data-MoreThanTwoMasks.xlsx file (optional)")
    p.add_argument("--mini2-base-dir", type=str, required=False, default="",
                   help="Base directory for Data-MoreThanTwoMasks images/masks (optional)")
    p.add_argument("--output-csv", type=str, required=True,
                   help="Output path for unified CSV")
    p.add_argument("--outdir", type=str, default="DATASET",
                   help="Output directory for processed images and masks (default: DATASET)")
    
    # Preprocessing arguments
    preproc = p.add_argument_group('preprocessing options')
    preproc.add_argument("--avoid-sigmoid", action="store_false",
                        help="Avoid sigmoid (non-linear) normalization instead of linear")
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
        'use_sigmoid': args.avoid_sigmoid,
        'median_kernel': args.median_kernel,
        'clahe_tile_size': args.clahe_tile_size,
        'clahe_delta': args.clahe_delta
    }
    
    # Handle optional mini2 arguments
    mini2_excel = args.mini2_excel if args.mini2_excel else None
    mini2_base = args.mini2_base_dir if args.mini2_base_dir else None
    
    # Process datasets
    process_datasets(
        args.cbis_csv,
        args.mini_ddsm_excel,
        args.mini_ddsm_base_dir,
        mini2_excel,
        mini2_base,
        args.output_csv,
        args.outdir,
        preproc_args
    )
