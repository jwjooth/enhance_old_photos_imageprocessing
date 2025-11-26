# 🧪 TESTING GUIDE - Simple Step by Step

**Verifikasi project sudah berjalan dengan baik dan sempurna!**

---

## ✅ CHECKLIST TESTING

Ikuti step-by-step dari atas ke bawah. Setiap step harus ✅ PASS!

---

## 🔍 STEP 1: Verify Environment (2 menit)

### Check Python Version

```bash
python --version
# Harus: Python 3.11 atau lebih tinggi
```

✅ PASS jika: `Python 3.11.x` atau `3.12.x`

---

### Check Virtual Environment Active

```bash
# Windows: Lihat awal command line
(venv) C:\...

# macOS/Linux: Lihat awal command line
(venv) $
```

✅ PASS jika: Ada `(venv)` di awal

Jika tidak ada, activate:
```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

---

### Check Dependencies Installed

```bash
# Windows
pip list | findstr /R "opencv numpy streamlit pytest"

# macOS/Linux
pip list | grep -E "opencv|numpy|streamlit|pytest"
```

✅ PASS jika: Semua installed (show version numbers)

---

## 🔍 STEP 2: Verify Project Structure (1 menit)
```bash Windows (CMD)
dir app.py main.py README.md requirements.txt

dir modules utils input output docs tests

dir pytest.ini .streamlit
```

```bash macOs/Linux
# Check main files exist
ls -la app.py main.py README.md requirements.txt

# Check folders exist
ls -la modules/ utils/ input/ output/ docs/ tests/

# Check pytest config
ls -la pytest.ini .streamlit/
```

✅ PASS jika: Semua files & folders ada

---

## 🧪 STEP 3: Run Unit Tests (5 menit)

### Check pytest installed

```bash
pytest --version
# Harus: pytest 9.0.0 atau lebih tinggi
```

✅ PASS jika: Show version number

---

### Collect tests (lihat berapa test)

```bash
pytest --collect-only
# Harus: collected 37 items
```

✅ PASS jika: `collected 37 items`

Jika error "no section header", fix `pytest.ini` (see README)

---

### Run all tests

```bash
pytest -v
```

Tunggu processing...

✅ PASS jika:
```
=============== 37 passed in X.XXs ================
```

Jika ada FAILED, lihat error message dan debug.

---

### Check test coverage (optional)

```bash
pytest --cov=modules
```

✅ PASS jika: Coverage ~90%

---

## 🔍 STEP 4: Verify Imports (2 menit)

Test apakah semua modules bisa di-import:

```bash
# Test geometric
python -c "from modules.geometric import GeometricCorrection; print('✅ geometric')"

# Test filtering
python -c "from modules.filtering import NoiseFiltering; print('✅ filtering')"

# Test histogram
python -c "from modules.histogram import HistogramEqualization; print('✅ histogram')"

# Test image_analyzer
python -c "from modules.image_analyzer import ImageAnalyzer; print('✅ analyzer')"

# Test smart_enhancer
python -c "from modules.smart_enhancer import SmartEnhancer; print('✅ smart')"
```

✅ PASS jika: Semua print `✅` (no errors)

---

## 🔍 STEP 5: Test Web App (3 menit)

### Start Streamlit

```bash
streamlit run app.py
```

Tunggu sampai show:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://...
```

✅ PASS jika: No errors, browser opens

---

### Test Web Interface

1. **Upload Section**
   - ✅ Drag-drop area terlihat
   - ✅ Upload button berfungsi

2. **Tabs**
   - ✅ "Upload & Process" tab aktif
   - ✅ "Results" tab ada
   - ✅ "Analytics" tab ada
   - ✅ "Guide" tab ada

3. **Settings (Sidebar)**
   - ✅ Mode selector ada
   - ✅ Sliders berfungsi
   - ✅ Checkboxes dapat di-click

4. **Buttons**
   - ✅ "START PROCESSING" button terlihat

✅ PASS jika: Semua elements terlihat & berfungsi

---

### Test Single Image Enhancement

1. **Prepare test image**
   - Gunakan foto `.jpg` atau `.png` dari komputer Anda
   - Size: 500KB atau lebih kecil

2. **Upload**
   - Drag-drop atau click upload
   - Lihat file terdeteksi

3. **Process**
   - Select mode: Smart
   - Click "START PROCESSING"
   - Tunggu sampai selesai

4. **Verify Results**
   - ✅ Progress bar menunjukkan progress
   - ✅ Success message muncul
   - ✅ Bisa switch ke "Results" tab
   - ✅ Before-after comparison terlihat
   - ✅ Download button berfungsi

✅ PASS jika: Photo berhasil di-enhance & terlihat lebih baik

---

## 🔍 STEP 6: Test CLI Version (2 menit)

```bash
python main.py
```

Menu muncul:
```
🎯 Select Mode:
1. Single Image Enhancement
2. Batch Processing
```

✅ PASS jika: Menu terlihat

---

### Test single image (CLI)

1. **Pilih mode 1**
2. **Input path:** `input/sample.jpg` (atau foto Anda)
3. **Tunggu processing**
4. **Check output folder**
   - ✅ Hasil tersimpan di `output/final/`

✅ PASS jika: Image berhasil di-enhance

---

## 🔍 STEP 7: Test Configuration (2 menit)

### Check config file

```bash
# Windows
type config\settings.json

# macOs/Linux
cat config/settings.json
# atau buka dengan text editor
```

✅ PASS jika: JSON valid & readable

---

### Verify config values

```bash
python -c "from utils.config_manager import ConfigManager; c = ConfigManager.load_config(); print(c['filtering'])"
```

✅ PASS jika: Config values muncul

---

## 🔍 STEP 8: Test Logging (1 menit)

```bash
python -c "from utils.logger import setup_logger; logger = setup_logger('test'); logger.info('Test log'); print('✅ Logger OK')"
```

✅ PASS jika: Log message muncul dengan warna

---

## 🔍 STEP 9: Test Performance (optional, 5 menit)

### Time single image processing

```bash
# Create test image
python -c "import cv2, numpy as np; img = np.ones((300,300,3), dtype=np.uint8)*128; cv2.imwrite('test_img.jpg', img)"

# Time it
time python -c "from modules.smart_enhancer import SmartEnhancer; import cv2; e = SmartEnhancer(); img = cv2.imread('test_img.jpg'); e.enhance(img)"
```

✅ PASS jika: Selesai dalam < 5 detik

---

## 🔍 STEP 10: Test Batch Processing (5 menit)

### Create test images

```bash
# Buat 3 test images di input/
# Copy foto Anda ke: input/test1.jpg, input/test2.jpg, input/test3.jpg
```

### Run batch via web

1. Open Streamlit: `streamlit run app.py`
2. Upload 3 images
3. Select Smart mode
4. Click "START PROCESSING"
5. Wait...

✅ PASS jika:
- ✅ All 3 processed successfully
- ✅ Results tab show 3 images
- ✅ Can download all

---

## 📊 SUMMARY CHECKLIST

```
✅ Step 1: Environment verified
✅ Step 2: Project structure correct
✅ Step 3: 37 tests passed
✅ Step 4: All imports working
✅ Step 5: Web app loads
✅ Step 6: Single image enhanced (web)
✅ Step 7: CLI working
✅ Step 8: Configuration loaded
✅ Step 9: Logging active
✅ Step 10: Batch processing works (optional)
```

---

## 🎯 FINAL VERIFICATION

Run this command untuk final check:

```bash
echo "=== PROJECT STATUS ===" && \
echo "✅ Python: $(python --version)" && \
echo "✅ Pytest: $(pytest --version)" && \
echo "✅ Tests: $(pytest --collect-only -q | tail -1)" && \
echo "✅ Imports: OK" && \
python -c "from modules.smart_enhancer import SmartEnhancer; print('✅ All modules loaded')" && \
echo "" && \
echo "🎉 PROJECT READY!"
```

✅ PASS jika: Semua show OK

---

## 🚀 NEXT STEPS

Jika semua test PASS:

1. ✅ Project fully functional
2. ✅ Ready for production
3. ✅ Can process photos
4. ✅ Can deploy (Docker, etc.)

---

## 🔧 TROUBLESHOOTING

| Error | Solution |
|-------|----------|
| pytest: command not found | `pip install pytest` |
| 37 items not collected | Add `tests/test_all.py` file |
| Import error | Check `modules/` folder exists |
| Web app won't start | `pip install streamlit` |
| Processing error | Check input image format |

---

## 📝 NOTES

- Semua test harus PASS
- Jika ada 1 FAIL, debug sebelum lanjut
- Test time: ~20 menit (first time)
- Setelah semua OK → project siap digunakan!

---

**Sudah siap? Start testing sekarang! 🚀**

`pytest -v` → Harus show `37 passed`