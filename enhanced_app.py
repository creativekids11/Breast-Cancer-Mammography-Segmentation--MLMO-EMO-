"""
Enhanced Flask Web UI for Breast Cancer Tumor Detection
Optimized for single-image processing with maximum accuracy
"""

from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image
import time

# Import the fixed implementation
from paper_exact_mlmo_emo import PaperSegmentationModel, PaperEvaluationMetrics

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize model with ENHANCED settings for single-image processing
# More computation power = better results for one image
model = PaperSegmentationModel(
    population_size=50,      # 5x larger population for better search
    max_iterations=100,      # More iterations for accuracy
    local_search_prob=0.8,   # More local refinement
    local_search_iterations=5,
    l1_weight=1e-5,
    use_mixed_precision=False,  # Use full precision for UI
    use_gpu=True,
    batch_size=32
)

metrics_calculator = PaperEvaluationMetrics()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'dcm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_overlay_image(original, mask):
    """Create visualization with tumor highlighted in red."""
    # Ensure images are same size
    if mask.shape != original.shape:
        mask = cv2.resize(mask, (original.shape[1], original.shape[0]), 
                         interpolation=cv2.INTER_NEAREST)
    
    # Convert grayscale to RGB
    if len(original.shape) == 2:
        original_rgb = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
    else:
        original_rgb = original.copy()
    
    # Create red overlay where mask == 1
    overlay = original_rgb.copy()
    overlay[mask == 1] = [255, 0, 0]  # Red for tumor
    
    # Blend: 60% original, 40% overlay for semi-transparent effect
    result = cv2.addWeighted(original_rgb, 0.6, overlay, 0.4, 0)
    
    # Draw contours for better visibility
    contours, _ = cv2.findContours((mask * 255).astype(np.uint8), 
                                   cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (255, 255, 0), 2)  # Yellow border
    
    return result

def advanced_tumor_detection(image, mask=None, method='otsu'):
    """
    Enhanced tumor detection with multiple strategies.
    Uses ensemble approach for single-image accuracy.
    """
    start_time = time.time()
    
    print(f"\n{'='*60}")
    print("ENHANCED TUMOR DETECTION")
    print(f"{'='*60}")
    print(f"Image shape: {image.shape}")
    print(f"Method: {method}")
    
    # Strategy 1: Standard segmentation
    print("\n[1/3] Running standard segmentation...")
    results1 = model.segment_image(image, method=method, num_thresholds=1)
    seg1 = results1['segmented']
    ratio1 = np.count_nonzero(seg1) / seg1.size
    print(f"      Tumor region: {ratio1*100:.4f}%")
    
    # Strategy 2: Try with inverted image (catches inverted contrast cases)
    print("[2/3] Running inverted segmentation...")
    inverted_image = 255 - image
    results2 = model.segment_image(inverted_image, method=method, num_thresholds=1)
    seg2 = results2['segmented']
    ratio2 = np.count_nonzero(seg2) / seg2.size
    print(f"      Tumor region: {ratio2*100:.4f}%")
    
    # Strategy 3: Adaptive histogram equalization (alternative preprocessing)
    print("[3/3] Running adaptive preprocessing...")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equalized = clahe.apply(image)
    results3 = model.segment_image(equalized, method=method, num_thresholds=1)
    seg3 = results3['segmented']
    ratio3 = np.count_nonzero(seg3) / seg3.size
    print(f"      Tumor region: {ratio3*100:.4f}%")
    
    # Select best result (closest to expected tumor size 0.1%-3%)
    target_ratio = 0.01  # 1% ideal
    scores = [
        abs(ratio1 - target_ratio),
        abs(ratio2 - target_ratio),
        abs(ratio3 - target_ratio)
    ]
    
    best_idx = np.argmin(scores)
    strategies = ['Standard', 'Inverted', 'Adaptive']
    
    if best_idx == 0:
        final_seg = seg1
        final_results = results1
    elif best_idx == 1:
        final_seg = seg2
        final_results = results2
    else:
        final_seg = seg3
        final_results = results3
    
    print(f"\n{'='*60}")
    print(f"BEST STRATEGY: {strategies[best_idx]}")
    print(f"Final tumor region: {np.count_nonzero(final_seg)/final_seg.size*100:.4f}%")
    print(f"Processing time: {time.time() - start_time:.2f}s")
    print(f"{'='*60}\n")
    
    # Calculate metrics if ground truth provided
    if mask is not None:
        # Normalize mask
        if mask.max() > 1:
            mask = (mask > 127).astype(np.uint8)
        mask = np.clip(mask, 0, 1)
        
        metrics = metrics_calculator.calculate_metrics(final_seg, mask)
    else:
        metrics = None
    
    return {
        'segmented': final_seg,
        'strategy_used': strategies[best_idx],
        'all_strategies': {
            'standard': {'seg': seg1, 'ratio': ratio1},
            'inverted': {'seg': seg2, 'ratio': ratio2},
            'adaptive': {'seg': seg3, 'ratio': ratio3}
        },
        'thresholds': final_results['thresholds'],
        'metrics': metrics,
        'processing_time': time.time() - start_time
    }

@app.route('/')
def index():
    return render_template('enhanced_index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Check if image file is provided
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(image_file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Load image
        filename = secure_filename(image_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(filepath)
        
        # Read image
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return jsonify({'error': 'Could not read image'}), 400
        
        # Check for ground truth mask (optional)
        mask = None
        if 'mask' in request.files and request.files['mask'].filename != '':
            mask_file = request.files['mask']
            mask_filename = secure_filename(mask_file.filename)
            mask_filepath = os.path.join(app.config['UPLOAD_FOLDER'], mask_filename)
            mask_file.save(mask_filepath)
            mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
        
        # Get method from form
        method = request.form.get('method', 'otsu')
        
        # Run enhanced detection
        results = advanced_tumor_detection(image, mask, method)
        
        # Create visualization
        overlay = create_overlay_image(image, results['segmented'])
        
        # Convert images to base64 for display
        def img_to_base64(img):
            if len(img.shape) == 2:  # Grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            _, buffer = cv2.imencode('.png', img)
            return base64.b64encode(buffer).decode('utf-8')
        
        response_data = {
            'success': True,
            'original_image': img_to_base64(image),
            'segmented_image': img_to_base64(results['segmented'] * 255),
            'overlay_image': img_to_base64(overlay),
            'strategy_used': results['strategy_used'],
            'tumor_percentage': f"{np.count_nonzero(results['segmented'])/results['segmented'].size*100:.4f}",
            'tumor_pixels': int(np.count_nonzero(results['segmented'])),
            'total_pixels': int(results['segmented'].size),
            'threshold': float(results['thresholds'][0]),
            'processing_time': f"{results['processing_time']:.2f}",
            'image_shape': image.shape
        }
        
        # Add metrics if available
        if results['metrics']:
            response_data['metrics'] = {
                'dice_coefficient': f"{results['metrics']['dice_coefficient']:.4f}",
                'jaccard_coefficient': f"{results['metrics']['jaccard_coefficient']:.4f}",
                'sensitivity': f"{results['metrics']['sensitivity']*100:.2f}",
                'specificity': f"{results['metrics']['specificity']*100:.2f}",
                'accuracy': f"{results['metrics']['accuracy']*100:.2f}",
                'true_positives': int(results['metrics']['TP']),
                'false_positives': int(results['metrics']['FP']),
                'false_negatives': int(results['metrics']['FN']),
                'true_negatives': int(results['metrics']['TN'])
            }
        
        # Add all strategies info
        response_data['all_strategies'] = {
            'standard': f"{results['all_strategies']['standard']['ratio']*100:.4f}%",
            'inverted': f"{results['all_strategies']['inverted']['ratio']*100:.4f}%",
            'adaptive': f"{results['all_strategies']['adaptive']['ratio']*100:.4f}%"
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': True})

if __name__ == '__main__':
    print("\n" + "="*60)
    print("ENHANCED TUMOR DETECTION UI")
    print("="*60)
    print("Features:")
    print("  ✓ 3 detection strategies (ensemble approach)")
    print("  ✓ Automatic best-strategy selection")
    print("  ✓ High-accuracy mode (50 population, 100 iterations)")
    print("  ✓ Real-time metrics with ground truth")
    print("  ✓ Visual overlay with contours")
    print("="*60)
    print("\nStarting server on http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
