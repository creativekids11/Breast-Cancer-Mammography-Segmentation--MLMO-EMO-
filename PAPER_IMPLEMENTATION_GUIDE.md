# Exact Implementation of MLMO-EMO Paper

This implementation provides the **exact methodology** described in the paper "Segmentation of Breast Masses in Mammogram Image Using Multilevel Multiobjective Electromagnetism-Like Optimization Algorithm" (2022).

**⚠️ Note:** The paper has been retracted, but the implementation follows the described methodology exactly.

## Paper Methodology Overview

The paper describes a traditional computer vision approach (not deep learning) with the following components:

### 1. Image Collection (Section 3.1)
- **DDSM Dataset:** 2620 mammographic images with pixel-level ground truth
- **MIAS Dataset:** 322 digitalized mammogram images (1024×1024 pixels)
- Both datasets include abnormal truth marking locations

### 2. Image Denoising/Preprocessing (Section 3.2)

#### 2.1 Normalization (Equation 1)
```
I_norm = (I - Min) / (Max - Min) × 255
```
- Converts pixel values to range [0, 255]

#### 2.2 Sigmoid Normalization (Equation 2)  
```
I_sigmoid = 1 / (1 + exp(-(I - β)/α))
```
- α: width parameter (default: 50.0)
- β: center parameter (default: 127.0)

#### 2.3 Median Filter (Equation 3)
- Nonlinear filter to eliminate noise
- Preserves edges while removing noise
- Default kernel size: 3×3

#### 2.4 CLAHE Enhancement (Equations 4-8)
**Clip Limit Calculation (Equation 4):**
```
Clip_limit = δ × M × N / L
```
- δ: contrast factor (default: 2.0)
- M, N: contextual region dimensions
- L: number of histogram bins (256)

### 3. Segmentation using EML Algorithm (Section 3.3)

#### 3.1 Optimization Problem (Equation 9)
```
min f(x) = Rn(x)
subject to: x ∈ V
```

#### 3.2 Feasible Search Space (Equation 10)
```
V = {x ∈ R^n : li ≤ xi ≤ ui}
```
- Search space for threshold values: [1, 254]

#### 3.3 EML Algorithm Steps
1. **Population Initialization:** 50 particles in search space
2. **Force Calculation:** Electromagnetic attraction/repulsion
3. **Position Update:** Based on total forces (Equation 11)
4. **Local Search:** Probability-based improvement
5. **Iterations:** 100 iterations total

#### 3.4 Thresholding Methods

**Bilevel Thresholding (Equation 12):**
```
G1 if pi ≤ th
G2 if pi > th
```

**Multilevel Thresholding (Equation 13):**
```
Gi if thi-1 < pi ≤ thi
```

**Objective Functions:**
- **OTSU:** Minimizes within-class variance
- **Kapur:** Maximizes entropy-based separability

### 4. Template Matching
- Validates segmentation against ground truth
- Uses correlation-based matching
- Calculates overlap metrics (Jaccard, Dice)

## Implementation Files

### 1. `paper_exact_mlmo_emo.py`
**Main implementation file containing:**

#### Classes:
- `PaperPreprocessor`: Exact preprocessing pipeline
- `ElectromagnetismLikeOptimizer`: EML algorithm implementation
- `PaperSegmentationModel`: Complete segmentation pipeline
- `PaperEvaluationMetrics`: Evaluation metrics (equations 14-18)

#### Key Methods:
```python
# Preprocessing
preprocessor = PaperPreprocessor()
results = preprocessor.preprocess_mammogram(image)

# Segmentation
model = PaperSegmentationModel()
segmentation = model.segment_image(image, method='otsu')

# Evaluation
metrics = PaperEvaluationMetrics.calculate_metrics(segmented, ground_truth)
```

### 2. `paper_exact_training.py`
**Training/evaluation script:**

#### Features:
- Dataset processing with CSV input
- Both OTSU and Kapur methods
- Batch processing with progress tracking
- Detailed results and statistics
- Comparison with paper results

#### Usage:
```bash
# Process dataset with OTSU method
python paper_exact_training.py --csv-path dataset.csv --method otsu

# Process limited number of images
python paper_exact_training.py --csv-path dataset.csv --method kapur --max-images 100

# Save detailed results
python paper_exact_training.py --csv-path dataset.csv --method otsu --save-detailed
```

## Evaluation Metrics (Equations 14-18)

The implementation calculates exact metrics as defined in the paper:

### Equation 14 - Jaccard Coefficient
```
J = TP / (TP + FP + FN)
```

### Equation 15 - Dice Coefficient  
```
D = (2 × TP) / (2 × TP + FP + FN)
```

### Equation 16 - Sensitivity (Recall)
```
Sen = TP / (TP + FN)
```

### Equation 17 - Specificity
```
Spe = TN / (TN + FP)  
```

### Equation 18 - Accuracy
```
Acc = (TP + TN) / (TP + TN + FP + FN)
```

Where:
- TP: True Positive
- TN: True Negative  
- FP: False Positive
- FN: False Negative

## Paper Results Comparison

### DDSM Dataset Results (Paper):
- **Sensitivity:** 92.3% (Benign: 92.2%, Malignant: 92.58%)
- **Specificity:** 99.21% (Benign: 99.12%, Malignant: 99.31%)
- **Accuracy:** 98.68% (Benign: 98.65%, Malignant: 98.72%)

### MIAS Dataset Results (Paper):
- **Sensitivity:** 92.11%  
- **Specificity:** 99.45%
- **Accuracy:** 98.93%

## Algorithm Parameters

### EML Optimization Parameters:
- **Population Size:** 50 particles
- **Max Iterations:** 100
- **Local Search Probability:** 0.8
- **Force Constant:** 1.0
- **Learning Rate:** 0.1 (decreasing)

### CLAHE Parameters:
- **Tile Size:** 8×8 pixels
- **Contrast Factor (δ):** 2.0
- **Histogram Bins (L):** 256

### Filter Parameters:
- **Median Filter:** 3×3 kernel
- **Normalization:** Min-max to [0, 255]

## Usage Example

```python
from paper_exact_mlmo_emo import PaperSegmentationModel, PaperEvaluationMetrics
import cv2

# Load mammogram and ground truth
image = cv2.imread('mammogram.png', cv2.IMREAD_GRAYSCALE)
ground_truth = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE)

# Initialize model
model = PaperSegmentationModel()

# Segment using OTSU method
results = model.segment_image(image, method='otsu', num_thresholds=1)

# Template matching
matching = model.template_matching(results['segmented'], ground_truth)

# Calculate metrics
metrics = PaperEvaluationMetrics.calculate_metrics(results['segmented'], ground_truth)

print(f"Jaccard: {metrics['jaccard_coefficient']:.4f}")
print(f"Dice: {metrics['dice_coefficient']:.4f}")  
print(f"Sensitivity: {metrics['sensitivity']:.4f}")
print(f"Specificity: {metrics['specificity']:.4f}")
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

## Key Differences from Original Code

### Original (Deep Learning Approach):
- PyTorch-based neural networks
- Encoder-decoder architecture  
- Gradient-based optimization
- End-to-end learning

### Paper Implementation (Traditional CV):
- Classical image processing
- Electromagnetism-like optimization
- Threshold-based segmentation
- Multi-objective optimization

## Integration with Existing Codebase

The paper implementation can be used alongside the existing deep learning approach:

1. **For comparison studies**
2. **For validating results**  
3. **For understanding classical methods**
4. **For hybrid approaches**

Both implementations share the same dataset processing pipeline and can be evaluated using the same metrics.

## Running the Implementation

### Prerequisites:
```bash
pip install opencv-python numpy matplotlib scikit-learn scipy scikit-image
```

### Quick Test:
```python
from paper_exact_mlmo_emo import demonstrate_paper_implementation
demonstrate_paper_implementation()
```

### Full Dataset Processing:
```bash
# Ensure you have the dataset CSV file
python paper_exact_training.py --csv-path unified_segmentation_dataset.csv --method otsu
```

This implementation provides a complete, exact reproduction of the paper's methodology for research and comparison purposes.