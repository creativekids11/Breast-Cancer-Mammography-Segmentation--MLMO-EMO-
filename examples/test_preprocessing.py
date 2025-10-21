"""
Test script to demonstrate the comprehensive image preprocessing pipeline
for mammographic images based on research paper equations.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from dataset_process import (
    linear_normalization,
    sigmoid_normalization,
    median_filtering,
    apply_clahe,
    preprocess_image,
    calculate_clahe_clip_limit
)


def create_test_image():
    """Create a synthetic test mammographic image."""
    # Create a base noisy image
    img = np.random.normal(100, 50, (512, 512)).astype(np.float32)
    img = np.clip(img, 0, 255)
    
    # Add some structure (simulated lesion)
    center_y, center_x = 256, 256
    radius = 80
    y, x = np.ogrid[:512, :512]
    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    img[mask] = img[mask] + 50
    
    # Add some noise
    noise = np.random.normal(0, 20, (512, 512))
    img = img + noise
    img = np.clip(img, 0, 255)
    
    return img.astype(np.uint8)


def visualize_normalization_comparison():
    """Compare linear vs sigmoid normalization."""
    img = create_test_image()
    
    # Apply different normalizations
    linear_norm = linear_normalization(img, new_min=0, new_max=255)
    sigmoid_norm = sigmoid_normalization(img, alpha=50.0, new_min=0, new_max=255)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(linear_norm, cmap='gray')
    axes[1].set_title('Linear Normalization\n(Equation 1)')
    axes[1].axis('off')
    
    axes[2].imshow(sigmoid_norm, cmap='gray')
    axes[2].set_title('Sigmoid Normalization\n(Equation 2)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('normalization_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: normalization_comparison.png")
    plt.close()


def visualize_preprocessing_pipeline():
    """Visualize complete preprocessing pipeline."""
    img = create_test_image()
    
    # Step-by-step preprocessing
    linear_norm = linear_normalization(img, new_min=0, new_max=255).astype(np.uint8)
    after_median = median_filtering(linear_norm, kernel_size=5)
    after_clahe = apply_clahe(after_median, tile_grid_size=(8, 8), delta=0.01)
    
    # Also show sigmoid version
    sigmoid_norm = sigmoid_normalization(img, alpha=50.0, new_min=0, new_max=255).astype(np.uint8)
    sigmoid_median = median_filtering(sigmoid_norm, kernel_size=5)
    sigmoid_clahe = apply_clahe(sigmoid_median, tile_grid_size=(8, 8), delta=0.01)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Linear pipeline
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Original Image', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(linear_norm, cmap='gray')
    axes[0, 1].set_title('1. Linear Normalization\n(Equation 1)', fontsize=12)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(after_median, cmap='gray')
    axes[0, 2].set_title('2. Median Filtering\n(Equation 3)', fontsize=12)
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(after_clahe, cmap='gray')
    axes[0, 3].set_title('3. CLAHE\n(Equations 4-8)', fontsize=12)
    axes[0, 3].axis('off')
    
    # Sigmoid pipeline
    axes[1, 0].imshow(img, cmap='gray')
    axes[1, 0].set_title('Original Image', fontsize=12)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(sigmoid_norm, cmap='gray')
    axes[1, 1].set_title('1. Sigmoid Normalization\n(Equation 2)', fontsize=12)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(sigmoid_median, cmap='gray')
    axes[1, 2].set_title('2. Median Filtering\n(Equation 3)', fontsize=12)
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(sigmoid_clahe, cmap='gray')
    axes[1, 3].set_title('3. CLAHE\n(Equations 4-8)', fontsize=12)
    axes[1, 3].axis('off')
    
    fig.suptitle('Comprehensive Preprocessing Pipeline\nTop: Linear | Bottom: Sigmoid', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('preprocessing_pipeline.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: preprocessing_pipeline.png")
    plt.close()


def test_clahe_clip_limit():
    """Test CLAHE clip limit calculation with different parameters."""
    print("\n" + "="*60)
    print("CLAHE Clip Limit Calculation (Equation 4)")
    print("="*60)
    
    # Test different configurations
    test_configs = [
        {"M": 64, "N": 64, "L": 256, "delta": 0.01},
        {"M": 64, "N": 64, "L": 256, "delta": 0.05},
        {"M": 128, "N": 128, "L": 256, "delta": 0.01},
        {"M": 32, "N": 32, "L": 256, "delta": 0.001},
    ]
    
    for config in test_configs:
        M, N, L, delta = config["M"], config["N"], config["L"], config["delta"]
        clip_limit = calculate_clahe_clip_limit(M, N, L, delta)
        
        print(f"\nConfiguration:")
        print(f"  M (rows): {M}, N (cols): {N}, L (bins): {L}, δ: {delta}")
        print(f"  Formula: δ×M×N/L = {delta}×{M}×{N}/{L} = {(delta*M*N)/L:.4f}")
        print(f"  Clip Limit (nT): {clip_limit:.4f}")


def test_median_filtering():
    """Test median filtering with different kernel sizes."""
    img = create_test_image()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    kernel_sizes = [3, 5, 7, 9, 11, 13]
    
    for idx, ksize in enumerate(kernel_sizes):
        row = idx // 3
        col = idx % 3
        
        filtered = median_filtering(img, kernel_size=ksize)
        
        axes[row, col].imshow(filtered, cmap='gray')
        axes[row, col].set_title(f'Median Filter (kernel={ksize})', fontsize=12)
        axes[row, col].axis('off')
    
    fig.suptitle('Median Filtering with Different Kernel Sizes\n(Equation 3)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('median_filtering_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: median_filtering_comparison.png")
    plt.close()


def test_preprocessing_variants():
    """Test preprocessing with different parameter combinations."""
    img = create_test_image()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Different preprocessing variants
    variants = [
        {"use_sigmoid": False, "median_kernel": 3, "clahe_delta": 0.01, "title": "Linear, K=3, δ=0.01"},
        {"use_sigmoid": False, "median_kernel": 7, "clahe_delta": 0.01, "title": "Linear, K=7, δ=0.01"},
        {"use_sigmoid": False, "median_kernel": 5, "clahe_delta": 0.02, "title": "Linear, K=5, δ=0.02"},
        {"use_sigmoid": True, "median_kernel": 3, "clahe_delta": 0.01, "title": "Sigmoid, K=3, δ=0.01"},
        {"use_sigmoid": True, "median_kernel": 7, "clahe_delta": 0.01, "title": "Sigmoid, K=7, δ=0.01"},
        {"use_sigmoid": True, "median_kernel": 5, "clahe_delta": 0.02, "title": "Sigmoid, K=5, δ=0.02"},
    ]
    
    for idx, variant in enumerate(variants):
        row = idx // 3
        col = idx % 3
        
        processed = preprocess_image(
            img,
            use_sigmoid=variant["use_sigmoid"],
            median_kernel=variant["median_kernel"],
            clahe_delta=variant["clahe_delta"]
        )
        
        axes[row, col].imshow(processed, cmap='gray')
        axes[row, col].set_title(variant["title"], fontsize=11)
        axes[row, col].axis('off')
    
    fig.suptitle('Preprocessing with Different Parameter Combinations', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('preprocessing_variants.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: preprocessing_variants.png")
    plt.close()


def main():
    print("="*60)
    print("Mammographic Image Preprocessing Test")
    print("Based on Research Paper Equations (1)-(8)")
    print("="*60)
    
    print("\n[1/6] Testing normalization techniques...")
    visualize_normalization_comparison()
    
    print("\n[2/6] Testing complete preprocessing pipeline...")
    visualize_preprocessing_pipeline()
    
    print("\n[3/6] Testing CLAHE clip limit calculations...")
    test_clahe_clip_limit()
    
    print("\n[4/6] Testing median filtering...")
    test_median_filtering()
    
    print("\n[5/6] Testing preprocessing variants...")
    test_preprocessing_variants()
    
    print("\n[6/6] Testing full preprocessing function...")
    img = create_test_image()
    
    # Test with linear normalization
    processed_linear = preprocess_image(img, use_sigmoid=False, median_kernel=5, clahe_delta=0.01)
    print(f"  - Linear preprocessing: {img.shape} -> {processed_linear.shape}")
    print(f"    Input range: [{img.min()}, {img.max()}]")
    print(f"    Output range: [{processed_linear.min()}, {processed_linear.max()}]")
    
    # Test with sigmoid normalization
    processed_sigmoid = preprocess_image(img, use_sigmoid=True, median_kernel=5, clahe_delta=0.01)
    print(f"  - Sigmoid preprocessing: {img.shape} -> {processed_sigmoid.shape}")
    print(f"    Input range: [{img.min()}, {img.max()}]")
    print(f"    Output range: [{processed_sigmoid.min()}, {processed_sigmoid.max()}]")
    
    print("\n" + "="*60)
    print("✅ All tests completed successfully!")
    print("="*60)
    print("\nGenerated visualizations:")
    print("  1. normalization_comparison.png")
    print("  2. preprocessing_pipeline.png")
    print("  3. median_filtering_comparison.png")
    print("  4. preprocessing_variants.png")
    print("\nThese visualizations demonstrate:")
    print("  - Equation (1): Linear Normalization")
    print("  - Equation (2): Sigmoid (Non-linear) Normalization")
    print("  - Equation (3): Median Filtering")
    print("  - Equations (4)-(8): CLAHE with clip limit calculation")


if __name__ == "__main__":
    main()
