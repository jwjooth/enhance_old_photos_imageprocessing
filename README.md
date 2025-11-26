# 🖼️ Old Photo Enhancement

**AI-Powered Professional Photo Restoration System**

Restore dan enhance old family photos menggunakan Computer Vision & AI dengan otomatis!

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Fitur Utama

### 🤖 Smart Enhancement (AI Auto-Detect)
- Otomatis analyze kondisi setiap foto
- Intelligent parameter recommendation
- Per-photo customization
- Detailed analysis report

### 📊 Advanced Features
- Batch processing (multiple photos)
- Before-after comparison
- Multiple export formats (JPG, PNG)
- Performance analytics
- Session management

### 🎨 Beautiful Web Interface
- Professional Streamlit UI
- Drag-and-drop upload
- Real-time processing
- Interactive dashboard

### ⚡ Production Ready
- 41+ unit tests
- Complete documentation
- Docker support
- Professional logging

---

## 🚀 Quick Start

### 1. Installation (5 menit)

```bash
# Clone repository
git clone https://github.com/jwjooth/enhance_old_photos_imageprocessing
cd FINAL_PROJECT 

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run App (2 menit)

```bash
# Streamlit Web App (Recommended)
streamlit run app.py

# Opens: http://localhost:8501
```

### 3. Upload & Process (1 menit)

1. Upload photo (drag-drop)
2. Select mode (Smart = auto-detect)
3. Click "START PROCESSING"
4. Download result

---

## 📁 Project Structure

```
old-photo-enhancement/
│
├── 📄 app.py                      ← Main Streamlit app
├── 📄 main.py                     ← CLI version
├── 📄 README.md                   ← This file
├── 📄 requirements.txt
│
├── 📂 config/
│   └── settings.json              ← Configuration
│
├── 📂 modules/                    ← 3 Core Enhancement Methods
│   ├── geometric.py               ← Step 1: Straighten
│   ├── filtering.py               ← Step 2: Clean
│   ├── histogram.py               ← Step 3: Brighten
│   ├── image_analyzer.py          ← Auto-detection
│   └── smart_enhancer.py          ← Smart pipeline
│
├── 📂 utils/                      ← Helper functions
│   ├── image_loader.py
│   ├── config_manager.py
│   ├── logger.py
│   └── metrics.py
│
├── 📂 input/                      ← Add your photos here
├── 📂 output/                     ← Enhanced photos saved here
│   ├── geometric/
│   ├── filtered/
│   ├── histogram/
│   ├── final/
│   └── comparisons/
│
├── 📂 tests/                      ← Unit tests
│   ├── __init__.py
│   ├── conftest.py
│   └── test_all.py
│
├── 📂 docs/                       ← Documentation
│   ├── SETUP.md
│   ├── USAGE.md
│   ├── API.md
│   └── EXAMPLES.md
│
└── 📂 .streamlit/
    └── config.toml                ← Streamlit config
```

---

## 🎯 How to Use

### Mode 1: Smart Enhancement (Recommended)

Best for: Batch processing, unknown photos, automatic optimization

```
1. Upload photos
2. Select "🤖 Smart (AI Auto-Detect)" mode
3. Click "START PROCESSING"
4. View analysis report (optional)
5. Download results
```

**Benefit:** Sistem otomatis analyze & recommend optimal settings!

### Mode 2: Manual Enhancement

Best for: Custom settings, specific requirements, learning

```
1. Upload photos
2. Select "⚙️ Manual (Custom Settings)" mode
3. Adjust sliders:
   - Filter method (bilateral, nlm, median, gaussian)
   - Filter strength (0.5 - 2.0)
   - Histogram clip limit (1.0 - 4.0)
4. Click "START PROCESSING"
5. Download results
```

---

## 3️⃣ Enhancement Pipeline

```
Original Photo
    ↓
Step 1: GEOMETRIC CORRECTION
├─ Auto-detect rotation
├─ Straighten if needed
└─ Output: Aligned photo
    ↓
Step 2: NOISE FILTERING
├─ Detect noise level
├─ Choose optimal filter
└─ Output: Clean photo
    ↓
Step 3: HISTOGRAM ENHANCEMENT
├─ Analyze brightness
├─ Apply CLAHE
├─ Restore colors (if faded)
└─ Output: Enhanced photo
    ↓
✨ FINAL RESULT
```

---

## 🧪 Testing & Verification

See: **TESTING_GUIDE.md** in root folder

Quick test:
```bash
# Run all tests
pytest -v

# Should show: 41 passed
```

---

## 📚 Documentation

- **[SETUP.md](docs/SETUP.md)** - Installation guide
- **[USAGE.md](docs/USAGE.md)** - User guide
- **[API.md](docs/API.md)** - API reference
- **[EXAMPLES.md](docs/EXAMPLES.md)** - Code examples

---

## 🔧 Technologies

- **Python 3.11+** - Programming language
- **OpenCV 4.8** - Image processing
- **NumPy** - Numerical computing
- **Streamlit 1.28** - Web interface
- **scikit-image** - Advanced filters

---

## 📊 Key Metrics

- **41+ Unit Tests** - Comprehensive coverage
- **~90% Code Coverage** - Well-tested modules
- **4 Documentation Files** - Complete reference
- **2 Enhancement Modes** - Flexible & powerful
- **3 Core Algorithms** - Geometric, Filtering, Histogram

---

## 🎯 Use Cases

✅ Restore old family photos
✅ Batch process photo albums
✅ Digitize scanned documents
✅ Enhance faded pictures
✅ Fix rotated/skewed photos
✅ Remove noise & artifacts

---

## 🤖 Smart Enhancement Explained

### What it does:

For each photo:
1. **Analyze** - Check rotation, noise, brightness, contrast, blur, color
2. **Recommend** - Suggest optimal enhancement method
3. **Apply** - Execute enhancement automatically
4. **Report** - Show exactly what was done & why

### Example:

```
Photo 1 (Dark & Noisy):
→ Analysis: severe darkness, moderate noise
→ Recommend: Strong CLAHE + bilateral filter
→ Result: Bright & clean

Photo 2 (Rotated):
→ Analysis: 12° rotation, good condition
→ Recommend: Geometric correction only
→ Result: Straightened & aligned
```

Each photo gets **custom treatment** based on its condition!

---

## 📥 Batch Processing

Upload multiple photos at once:

```bash
# Upload 5 photos
1. Click upload area
2. Hold Ctrl (Cmd on Mac)
3. Select multiple files
4. Or drag-drop multiple

# Process all automatically
5. Click "START PROCESSING"
6. Wait for completion

# Download all results
7. View in Results tab
```

---

## ⚙️ Configuration

Edit `config/settings.json` to customize:

```json
{
  "geometric": {
    "auto_rotation": true,      // Enable rotation detection
    "angle_threshold": 5         // Min angle to fix
  },
  "filtering": {
    "method": "bilateral",       // Filter type
    "strength": 1.0              // 0.5-2.0
  },
  "histogram": {
    "method": "clahe",           // Enhancement method
    "clip_limit": 2.0            // 1.0-4.0
  }
}
```

---

## 📤 Export Options

- **JPG** - Smaller file, good quality
- **PNG** - Lossless, larger file
- **Adjustable Quality** - 70-100%
- **Comparison Images** - Before-after

---

## 🐛 Troubleshooting

### App won't start
```bash
pip install streamlit>=1.28.0
streamlit run app.py
```

### Port already in use
```bash
streamlit run app.py --server.port 8502
```

### Processing is slow
1. Reduce max_image_size in config
2. Use bilateral filter (faster)
3. Process smaller batches

### Result looks wrong
1. Check Analysis Report (Smart mode)
2. Try different settings (Manual mode)
3. Check input photo quality

---

## 🚀 Advanced Usage

### Programmatic Enhancement

```python
from modules.smart_enhancer import SmartEnhancer
import cv2

image = cv2.imread('photo.jpg')
enhancer = SmartEnhancer()
result = enhancer.enhance(image)

# Access results
print(f"Summary: {result['summary']}")
cv2.imwrite('enhanced.jpg', result['enhanced'])
```

### Batch Processing

```python
from modules.smart_enhancer import SmartEnhancer
from pathlib import Path

enhancer = SmartEnhancer()
images = {f.name: cv2.imread(str(f)) 
          for f in Path('input').glob('*.jpg')}

results = enhancer.batch_enhance(images)

# Save all
for name, result in results.items():
    cv2.imwrite(f'output/{name}', result['enhanced'])
```

See [EXAMPLES.md](docs/EXAMPLES.md) for more!

---

## 🧪 Testing

```bash
# Run all tests
pytest -v

# Expected: 41 passed

# With coverage
pytest --cov=modules

# Run specific test
pytest tests/test_all.py::TestGeometricCorrection -v
```

---

## 📝 Performance

- Single image: 1-5 seconds
- Batch (10 photos): 20-40 seconds
- Memory: ~2-3x image size
- GPU: Optional support

---

## 📞 Support

- **Issues?** Check [docs/](docs/) folder
- **Questions?** See [USAGE.md](docs/USAGE.md)
- **Code examples?** See [EXAMPLES.md](docs/EXAMPLES.md)

---

## 📄 License

MIT License - Free for personal & commercial use

---

## ✨ Credits

Made with ❤️ for preserving precious memories

**Technologies:**
- OpenCV team
- Streamlit team
- NumPy & SciPy teams
- Python community

---

## 🎉 Getting Started Now

```bash
# 1. Clone & setup (5 min)
git clone https://github.com/jwjooth/enhance_old_photos_imageprocessing
cd FINAL_PROJECT 
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate
pip install -r requirements.txt

# 2. Run app (1 min)
streamlit run app.py

# 3. Upload & process (1 min)
Open http://localhost:8501
Upload photo → Click Process → Download 

Done! 🎊
```

---

**Ready? Start now! → `streamlit run app.py`** 🚀