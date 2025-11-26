# 💡 Code Examples - Old Photo Enhancement

## Basic Examples

### Example 1: Simple Enhancement

\`\`\`python
import cv2
from modules.smart_enhancer import SmartEnhancer

# Load photo
image = cv2.imread('old_photo.jpg')

# Enhance
enhancer = SmartEnhancer()
result = enhancer.enhance(image)

# Save
cv2.imwrite('enhanced_photo.jpg', result['enhanced'])
print("✅ Done!")
\`\`\`

### Example 2: Batch Processing

\`\`\`python
import cv2
from pathlib import Path
from modules.smart_enhancer import SmartEnhancer

enhancer = SmartEnhancer()

# Load all images
input_folder = Path('input')
images = {}
for file in input_folder.glob('*.jpg'):
    images[file.name] = cv2.imread(str(file))

# Process batch
results = enhancer.batch_enhance(images)

# Save all
output_folder = Path('output/final')
for filename, result in results.items():
    cv2.imwrite(str(output_folder / f'enhanced_{filename}'), result['enhanced'])
\`\`\`

### Example 3: Analyze Before Enhancement

\`\`\`python
import cv2
from modules.image_analyzer import ImageAnalyzer
from modules.smart_enhancer import SmartEnhancer

image = cv2.imread('photo.jpg')

# Analyze
analyzer = ImageAnalyzer()
analysis = analyzer.analyze(image)

print("Image Conditions:")
print(f"- Brightness: {analysis['report']['image_quality']['brightness']}")
print(f"- Noise: {analysis['report']['image_quality']['noise']}")
print(f"- Contrast: {analysis['report']['image_quality']['contrast']}")

# Enhance
enhancer = SmartEnhancer()
result = enhancer.enhance(image)
print(f"\\nEnhancements: {result['summary']}")
\`\`\`

## Advanced Examples

### Example 4: Manual Enhancement with Custom Settings

\`\`\`python
import cv2
from modules.geometric import GeometricCorrection
from modules.filtering import NoiseFiltering
from modules.histogram import HistogramEqualization

image = cv2.imread('photo.jpg')

# Step 1: Geometric
geo = GeometricCorrection()
image = geo.correct(image, auto_rotation=True, angle_threshold=5)['image']

# Step 2: Filtering
filt = NoiseFiltering()
image = filt.apply_filter(image, 'bilateral', strength=1.2)['image']

# Step 3: Histogram
hist = HistogramEqualization()
image = hist.enhance(image, 'clahe', clip_limit=2.5)['image']
image = hist.color_balance(image)
image = hist.local_contrast_enhancement(image, strength=1.5)

# Save
cv2.imwrite('enhanced.jpg', image)
\`\`\`

### Example 5: Process with Progress Tracking

\`\`\`python
import cv2
from pathlib import Path
from modules.smart_enhancer import SmartEnhancer

enhancer = SmartEnhancer()
input_folder = Path('input')

files = list(input_folder.glob('*.jpg'))
total = len(files)

for idx, file in enumerate(files, 1):
    print(f"[{idx}/{total}] Processing {file.name}...")
    
    image = cv2.imread(str(file))
    result = enhancer.enhance(image)
    
    output_path = Path('output/final') / f'enhanced_{file.name}'
    cv2.imwrite(str(output_path), result['enhanced'])
    
    print(f"  ✅ Done: {result['summary']}")

print(f"\\n✅ All {total} photos processed!")
\`\`\`

### Example 6: Export Analysis Reports

\`\`\`python
import cv2
import json
from modules.smart_enhancer import SmartEnhancer

enhancer = SmartEnhancer()
image = cv2.imread('photo.jpg')
result = enhancer.enhance(image)

# Save enhanced image
cv2.imwrite('enhanced.jpg', result['enhanced'])

# Save report as JSON
report = {
    'filename': 'photo.jpg',
    'metrics': result['report']['metrics'],
    'enhancements': result['report']['enhancements_applied'],
    'summary': result['summary']
}

with open('report.json', 'w') as f:
    json.dump(report, f, indent=2)

print("✅ Image and report saved!")
\`\`\`

## Streamlit Integration

### Example 7: Custom Streamlit Widget

\`\`\`python
import streamlit as st
import cv2
from modules.smart_enhancer import SmartEnhancer

st.title("Photo Enhancer")

# Upload
file = st.file_uploader("Upload photo")

if file:
    # Read
    img_array = cv2.imdecode(...)
    
    # Process
    enhancer = SmartEnhancer()
    result = enhancer.enhance(img_array)
    
    # Display
    col1, col2 = st.columns(2)
    with col1:
        st.image(result['original'])
    with col2:
        st.image(result['enhanced'])
    
    # Download
    st.download_button(
        "Download",
        cv2.imencode('.jpg', result['enhanced'])[1].tobytes(),
        "enhanced.jpg"
    )
\`\`\`

## Testing Examples

### Example 8: Unit Test

\`\`\`python
import cv2
import numpy as np
from modules.smart_enhancer import SmartEnhancer

def test_enhancement():
    # Create test image
    image = np.ones((300, 300, 3), dtype=np.uint8) * 128
    
    # Enhance
    enhancer = SmartEnhancer()
    result = enhancer.enhance(image)
    
    # Verify
    assert result['enhanced'] is not None
    assert result['enhanced'].shape == image.shape
    assert 'summary' in result
    
    print("✅ Test passed!")

test_enhancement()
\`\`\`

## Performance Optimization

### Example 9: Batch with Threading

\`\`\`python
import cv2
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from modules.smart_enhancer import SmartEnhancer

def enhance_image(file_path):
    enhancer = SmartEnhancer()
    image = cv2.imread(str(file_path))
    result = enhancer.enhance(image)
    cv2.imwrite(str(Path('output/final') / f'enhanced_{file_path.name}'), result['enhanced'])
    return f"✅ {file_path.name}"

# Process in parallel
input_folder = Path('input')
files = list(input_folder.glob('*.jpg'))

with ThreadPoolExecutor(max_workers=4) as executor:
    results = executor.map(enhance_image, files)
    for result in results:
        print(result)
\`\`\`