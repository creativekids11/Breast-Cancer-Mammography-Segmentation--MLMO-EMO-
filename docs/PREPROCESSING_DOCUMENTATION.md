# Image Preprocessing Pipeline Documentation

## Overview

This document describes the comprehensive image preprocessing pipeline implemented in `dataset_process.py` based on research paper specifications. The pipeline implements three main techniques for enhancing mammographic images:

1. **Normalization** (Linear and Sigmoid)
2. **Median Filtering** (Noise Reduction)
3. **CLAHE** (Contrast Limited Adaptive Histogram Equalization)

---

## Mathematical Formulations

### 1. Linear Normalization (Equation 1)

**Formula:**
```
Normalization = (I - Min) × (newMax - newMin) / (Max - Min) + newMin
```

**Where:**
- `I` = Original mammographic image
- `Min` = Minimum pixel value (0 for 8-bit images)
- `Max` = Maximum pixel value (255 for 8-bit images)
- `newMin` = Target minimum (default: 0)
- `newMax` = Target maximum (default: 255)

**Implementation:**
```python
def linear_normalization(image, new_min=0.0, new_max=255.0):
    img_min = np.min(image)
    img_max = np.max(image)
    normalized = (image - img_min) * (new_max - new_min) / (img_max - img_min) + new_min
    return normalized
```

**Purpose:**
- Adjusts pixel value range
- Enhances contrast
- Standardizes intensity distribution

---

### 2. Sigmoid (Non-Linear) Normalization (Equation 2)

**Formula:**
```
Normalization = (newMax - newMin) × 1 / (1 + e^(-(I - β)/α)) + newMin
```

**Where:**
- `I` = Original mammographic image
- `β` = Centered pixel value (typically mean of image)
- `α` = Width of pixel value (controls steepness, default: 50.0)
- `newMin` = Target minimum (default: 0)
- `newMax` = Target maximum (default: 255)

**Implementation:**
```python
def sigmoid_normalization(image, alpha=50.0, beta=None, new_min=0.0, new_max=255.0):
    if beta is None:
        beta = np.mean(image)
    sigmoid_component = 1.0 / (1.0 + np.exp(-(image - beta) / alpha))
    normalized = (new_max - new_min) * sigmoid_component + new_min
    return normalized
```

**Purpose:**
- Non-linear transformation
- Better for images with high dynamic range
- Preserves relative intensity relationships
- Reduces effect of outliers

---

### 3. Median Filtering (Equation 3)

**Formula:**
```
median[A(x) + B(x)] ≠ median[A(x)] + median[B(x)]
```

**Where:**
- `A(x)` and `B(x)` = Two different mammogram images
- Median is calculated from neighborhood pixels

**Implementation:**
```python
def median_filtering(image, kernel_size=5):
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure odd kernel size
    filtered = cv2.medianBlur(image.astype(np.uint8), kernel_size)
    return filtered
```

**Properties:**
- Order statistics filter
- Better noise reduction than linear filters
- Preserves edges
- Non-linear operation (not distributive)

**Purpose:**
- Removes salt-and-pepper noise
- Preserves image structure
- Prepares image for contrast enhancement

---

### 4. CLAHE Clip Limit Calculation (Equation 4)

**Formula:**
```
nT = {
    1,              if δ×M×N/L < 1
    δ×M×N/L,        else
}
where 0 < δ ≤ 1
```

**Where:**
- `L` = Number of histogram bins (256 for 8-bit images)
- `M` = Rows in contextual region (tile height)
- `N` = Columns in contextual region (tile width)
- `δ` = User-defined contrast factor (0 < δ ≤ 1)
- `nT` = Clip limit threshold

**Implementation:**
```python
def calculate_clahe_clip_limit(M, N, L=256, delta=0.01):
    clip_value = (delta * M * N) / L
    if clip_value < 1:
        n_T = 1.0
    else:
        n_T = clip_value
    return n_T
```

**Purpose:**
- Determines histogram clipping threshold
- Prevents over-amplification of noise
- Controls contrast enhancement strength

---

### 5. Original Histogram Clipping (Equation 5)

**Formula:**
```
hk = {
    nT,     if nk ≥ nT
    nk,     else
}
k = 1, 2, ..., L-1
```

**Where:**
- `hk` = Clipped histogram height
- `nk` = Original contextual region histogram
- `nT` = Clip limit threshold
- `L` = Number of histogram bins

**Purpose:**
- Limits histogram bin heights
- Prevents excessive contrast in uniform regions
- Prepares for pixel redistribution

---

### 6. Total Clipped Pixels (Equation 6)

**Formula:**
```
nc = M×N - Σ(k=0 to L-1) hk
```

**Where:**
- `nc` = Total number of clipped pixels
- `M×N` = Total pixels in tile
- `Σhk` = Sum of all clipped histogram bins

**Purpose:**
- Calculates pixels to redistribute
- Ensures histogram renormalization
- Maintains total pixel count

---

### 7. Pixel Redistribution (Equation 7)

**Formula:**
```
nμ = nc / L = (M×N - Σ(k=0 to L-1) hk) / L
```

**Where:**
- `nμ` = Number of pixels distributed to each bin
- `nc` = Total clipped pixels
- `L` = Number of histogram bins

**Purpose:**
- Distributes clipped pixels equally
- Renormalizes histogram
- Ensures no bin exceeds clip limit

---

### 8. Renormalized Histogram (Equation 8)

**Formula:**
```
hk = {
    nT,         if nk + nμ ≥ nT
    nk + nμ,    otherwise
}
```

**Where:**
- `hk` = Final renormalized histogram
- `nk` = Original histogram value
- `nμ` = Redistribution increment
- `nT` = Clip limit

**Purpose:**
- Creates final histogram for equalization
- Applies redistributed pixels
- Maintains clip limit constraint

---

## Complete Preprocessing Pipeline

### Function Signature

```python
def preprocess_image(
    img: np.ndarray,
    use_sigmoid: bool = False,
    median_kernel: int = 5,
    clahe_tile_size: Tuple[int, int] = (8, 8),
    clahe_delta: float = 0.01
) -> np.ndarray
```

### Pipeline Steps

1. **Normalization**
   - Linear (default) or Sigmoid (if `use_sigmoid=True`)
   - Standardizes intensity range to [0, 255]

2. **Median Filtering**
   - Applies order statistics filter
   - Removes noise while preserving edges
   - Kernel size configurable (default: 5×5)

3. **CLAHE**
   - Divides image into tiles
   - Calculates and clips histograms
   - Redistributes clipped pixels
   - Applies adaptive equalization

### Usage Examples

#### Basic Usage (Default Parameters)
```python
from dataset_process import preprocess_image
import cv2

# Load image
img = cv2.imread('mammogram.png', cv2.IMREAD_GRAYSCALE)

# Apply preprocessing
processed = preprocess_image(img)

# Save result
cv2.imwrite('processed.png', processed)
```

#### Custom Parameters
```python
# Use sigmoid normalization with larger median kernel
processed = preprocess_image(
    img,
    use_sigmoid=True,
    median_kernel=7,
    clahe_tile_size=(16, 16),
    clahe_delta=0.02
)
```

#### Linear Normalization with Aggressive Contrast
```python
processed = preprocess_image(
    img,
    use_sigmoid=False,
    median_kernel=5,
    clahe_delta=0.05  # Higher contrast
)
```

---

## Command-Line Usage

### Basic Processing
```bash
python dataset_process.py \
  --cbis-csv data.csv \
  --mini-ddsm-excel DataWMask.xlsx \
  --mini-ddsm-base-dir ./mini-ddsm \
  --output-csv unified_dataset.csv
```

### With Sigmoid Normalization
```bash
python dataset_process.py \
  --cbis-csv data.csv \
  --mini-ddsm-excel DataWMask.xlsx \
  --mini-ddsm-base-dir ./mini-ddsm \
  --output-csv unified_dataset.csv \
  --use-sigmoid
```

### Custom Preprocessing Parameters
```bash
python dataset_process.py \
  --cbis-csv data.csv \
  --mini-ddsm-excel DataWMask.xlsx \
  --mini-ddsm-base-dir ./mini-ddsm \
  --output-csv unified_dataset.csv \
  --median-kernel 7 \
  --clahe-tile-size 16 16 \
  --clahe-delta 0.02
```

### View All Options
```bash
python dataset_process.py --help
```

---

## Parameter Tuning Guide

### Normalization Type

| Type | Use Case | Advantages |
|------|----------|------------|
| **Linear** | Standard images, balanced contrast | Simple, predictable, fast |
| **Sigmoid** | High dynamic range, outliers present | Non-linear, robust to outliers |

### Median Kernel Size

| Size | Effect | Use Case |
|------|--------|----------|
| **3×3** | Light smoothing | Minimal noise, preserve details |
| **5×5** | Moderate smoothing | Balanced (default) |
| **7×7** | Strong smoothing | Heavy noise, less detail critical |
| **9×9+** | Very strong | Extreme noise, low-frequency analysis |

### CLAHE Tile Size

| Size | Effect | Use Case |
|------|--------|----------|
| **4×4** | Fine-grained | Small local variations |
| **8×8** | Balanced | Standard (default) |
| **16×16** | Coarse-grained | Large uniform regions |
| **32×32+** | Very coarse | Whole-image trends |

### CLAHE Delta (δ)

| Range | Effect | Use Case |
|-------|--------|----------|
| **0.001-0.01** | Subtle | Preserve details, minimal artifacts |
| **0.01-0.03** | Moderate | Balanced enhancement (default: 0.01) |
| **0.03-0.1** | Strong | Dramatic contrast, potential artifacts |
| **>0.1** | Very strong | Extreme cases, high risk of artifacts |

---

## Testing

### Run Comprehensive Tests
```bash
python test_preprocessing.py
```

This generates visualizations demonstrating:
1. Linear vs Sigmoid normalization
2. Complete preprocessing pipeline
3. Median filtering with different kernels
4. Preprocessing variants with different parameters

### Output Files
- `normalization_comparison.png` - Equation (1) vs (2)
- `preprocessing_pipeline.png` - Full pipeline visualization
- `median_filtering_comparison.png` - Equation (3) demonstrations
- `preprocessing_variants.png` - Parameter combinations

---

## Implementation Notes

### Computational Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Linear Normalization | O(n) | Single pass over pixels |
| Sigmoid Normalization | O(n) | Single pass with exp() |
| Median Filtering | O(n×k²) | k = kernel size |
| CLAHE | O(n + t×L) | t = tiles, L = bins |

### Memory Requirements

- Input image: W×H bytes (grayscale)
- Intermediate buffers: ~3×(W×H) bytes
- CLAHE histograms: tiles×L×4 bytes

### Performance Tips

1. **Use smaller tile sizes** for faster CLAHE (but coarser adaptation)
2. **Smaller median kernels** reduce computation time
3. **Linear normalization** is faster than sigmoid
4. **Batch processing** amortizes initialization overhead

---

## References

Equations (1)-(8) are based on research paper specifications for mammographic image preprocessing:

1. **Normalization**: Standardizes intensity distribution
2. **Median Filtering**: Noise reduction with edge preservation
3. **CLAHE**: Adaptive contrast enhancement with clip limiting

---

## Troubleshooting

### Issue: Over-enhanced images
**Solution**: Reduce `clahe_delta` (try 0.005-0.01)

### Issue: Too much noise
**Solution**: Increase `median_kernel` (try 7 or 9)

### Issue: Loss of detail
**Solution**: 
- Reduce `median_kernel` (try 3)
- Use smaller `clahe_tile_size` (try 4×4)

### Issue: Uneven contrast
**Solution**:
- Increase `clahe_tile_size` for more uniform enhancement
- Try sigmoid normalization

### Issue: Processing too slow
**Solution**:
- Use linear instead of sigmoid normalization
- Reduce `median_kernel` size
- Increase `clahe_tile_size`

---

## Summary

This implementation provides a research-grade image preprocessing pipeline for mammographic images with:

✅ **Mathematical rigor**: All equations (1)-(8) implemented exactly as specified  
✅ **Flexibility**: Configurable parameters for different use cases  
✅ **Performance**: Optimized using OpenCV for speed  
✅ **Validation**: Comprehensive test suite with visualizations  
✅ **Documentation**: Detailed explanations and usage examples  

The preprocessing pipeline significantly improves image quality for downstream segmentation tasks while maintaining computational efficiency.
