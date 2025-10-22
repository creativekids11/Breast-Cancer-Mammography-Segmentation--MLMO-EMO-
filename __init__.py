"""
Paper Exact Implementation Package

This package contains the exact implementation of the methodology described in:
"Segmentation of Breast Masses in Mammogram Image Using Multilevel Multiobjective 
Electromagnetism-Like Optimization Algorithm"

Paper DOI: https://onlinelibrary.wiley.com/doi/10.1155/2022/8576768

The implementation follows the paper's methodology exactly:
1. Image preprocessing (Section 3.2) - normalization, CLAHE, median filter
2. EML optimization (Section 3.3) - electromagnetism-like optimization 
3. Thresholding (OTSU and Kapur methods)
4. Template matching and evaluation (equations 14-18)

Main Components:
- PaperPreprocessor: Exact preprocessing pipeline
- ElectromagnetismLikeOptimizer: EML algorithm implementation  
- PaperSegmentationModel: Complete segmentation model
- PaperEvaluationMetrics: Evaluation metrics as per paper equations

Usage:
    from paper_exact_implementation.paper_exact_mlmo_emo import PaperSegmentationModel
    
    model = PaperSegmentationModel()
    results = model.segment_image(image, method='otsu')
"""

# Import main classes for easy access
from .paper_exact_mlmo_emo import (
    PaperPreprocessor,
    ElectromagnetismLikeOptimizer,
    PaperSegmentationModel,
    PaperEvaluationMetrics
)

__version__ = "1.0.0"
__author__ = "Paper Implementation Team"

__all__ = [
    'PaperPreprocessor',
    'ElectromagnetismLikeOptimizer', 
    'PaperSegmentationModel',
    'PaperEvaluationMetrics'
]