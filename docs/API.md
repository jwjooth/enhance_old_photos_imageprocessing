# 🔌 API Reference - Old Photo Enhancement

## Module: smart_enhancer

### SmartEnhancer Class

\`\`\`python
from modules.smart_enhancer import SmartEnhancer

enhancer = SmartEnhancer()
\`\`\`

#### Method: enhance()

Auto-enhance single image.

**Syntax:**
\`\`\`python
result = enhancer.enhance(image)
\`\`\`

**Parameters:**
- `image` (np.ndarray): Input image (BGR format)

**Returns:**
\`\`\`python
{
    'original': np.ndarray,          # Original image
    'enhanced': np.ndarray,          # Enhanced image
    'analysis': dict,                # Analysis results
    'recommendations': dict,         # Enhancement recommendations
    'report': dict,                  # Detailed report
    'steps_applied': list,           # Steps with details
    'summary': str                   # Human-readable summary
}
\`\`\`

**Example:**
\`\`\`python
import cv2
from modules.smart_enhancer import SmartEnhancer

image = cv2.imread('photo.jpg')
enhancer = SmartEnhancer()
result = enhancer.enhance(image)

print(f"Summary: {result['summary']}")
cv2.imwrite('enhanced.jpg', result['enhanced'])
\`\`\`

#### Method: batch_enhance()

Enhance multiple images.

**Syntax:**
\`\`\`python
results = enhancer.batch_enhance(images)
\`\`\`

**Parameters:**
- `images` (dict): {filename: image_array}

**Returns:**
- dict: {filename: enhancement_result}

**Example:**
\`\`\`python
import cv2
from modules.smart_enhancer import SmartEnhancer

images = {
    'photo1.jpg': cv2.imread('photo1.jpg'),
    'photo2.jpg': cv2.imread('photo2.jpg')
}

enhancer = SmartEnhancer()
results = enhancer.batch_enhance(images)

for filename, result in results.items():
    cv2.imwrite(f'enhanced_{filename}', result['enhanced'])
\`\`\`

---

## Module: image_analyzer

### ImageAnalyzer Class

\`\`\`python
from modules.image_analyzer import ImageAnalyzer

analyzer = ImageAnalyzer()
\`\`\`

#### Method: analyze()

Analyze image conditions.

**Syntax:**
\`\`\`python
result = analyzer.analyze(image)
\`\`\`

**Returns:**
\`\`\`python
{
    'analysis': {
        'rotation': {...},       # Rotation info
        'noise': {...},         # Noise analysis
        'brightness': {...},    # Brightness analysis
        'contrast': {...},      # Contrast analysis
        'blur': {...},          # Blur analysis
        'color_fading': {...}   # Color fading info
    },
    'recommendations': {...},   # Enhancement recommendations
    'report': {...}             # Detailed report
}
\`\`\`

---

## Module: geometric

### GeometricCorrection Class

\`\`\`python
from modules.geometric import GeometricCorrection

geo = GeometricCorrection()
\`\`\`

#### Method: correct()

Auto-detect and correct rotation.

\`\`\`python
result = geo.correct(image, auto_rotation=True, angle_threshold=5)
enhanced_image = result['image']
angle = result['angle']
\`\`\`

#### Method: manual_rotate()

Rotate with specified angle.

\`\`\`python
rotated = geo.manual_rotate(image, angle=45)
\`\`\`

---

## Module: filtering

### NoiseFiltering Class

\`\`\`python
from modules.filtering import NoiseFiltering

filt = NoiseFiltering()
\`\`\`

#### Method: apply_filter()

Apply noise filter.

\`\`\`python
result = filt.apply_filter(image, filter_type='bilateral', strength=1.0)
filtered = result['image']
\`\`\`

**filter_type options:**
- `'bilateral'` - Edge-preserving (recommended)
- `'nlm'` - Non-local means
- `'median'` - Median filtering
- `'gaussian'` - Gaussian blur

#### Method: combined_filter()

Combine two filters.

\`\`\`python
filtered = filt.combined_filter(image, 'bilateral', 'nlm', strength=1.0)
\`\`\`

---

## Module: histogram

### HistogramEqualization Class

\`\`\`python
from modules.histogram import HistogramEqualization

hist = HistogramEqualization()
\`\`\`

#### Method: enhance()

Enhance contrast and brightness.

\`\`\`python
result = hist.enhance(image, method='clahe', clip_limit=2.0)
enhanced = result['image']
\`\`\`

**method options:**
- `'clahe'` - Adaptive (recommended)
- `'standard'` - Standard equalization
- `'multiscale'` - Multi-scale

#### Method: color_balance()

Restore faded colors.

\`\`\`python
balanced = hist.color_balance(image)
\`\`\`

#### Method: local_contrast_enhancement()

Enhance local contrast.

\`\`\`python
enhanced = hist.local_contrast_enhancement(image, strength=1.5)
\`\`\`

---

## Complete Example

\`\`\`python
import cv2
from modules.smart_enhancer import SmartEnhancer
from modules.image_analyzer import ImageAnalyzer

# Load image
image = cv2.imread('old_photo.jpg')

# Analyze
analyzer = ImageAnalyzer()
analysis = analyzer.analyze(image)
print(f"Analysis: {analysis['analysis']}")

# Enhance
enhancer = SmartEnhancer()
result = enhancer.enhance(image)

# Save
cv2.imwrite('enhanced.jpg', result['enhanced'])

# Print report
print(f"Summary: {result['summary']}")
for step in result['steps_applied']:
    print(f"- {step['step']}: {step['reason']}")
\`\`\`