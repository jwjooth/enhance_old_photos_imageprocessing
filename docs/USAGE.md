# 📖 User Guide - Old Photo Enhancement

## Quick Start

1. **Run the app**
   \`\`\`bash
   streamlit run app.py
   \`\`\`

2. **Upload photos**
   - Drag-drop into the upload area
   - Or click to browse

3. **Choose mode**
   - 🤖 Smart: AI auto-optimizes
   - ⚙️ Manual: Custom settings

4. **Process**
   - Click "START PROCESSING"
   - Wait for results

5. **Download**
   - View in Results tab
   - Download as JPG or PNG

## Modes Explained

### 🤖 Smart Mode

**What it does:**
- Auto-analyzes each photo
- Recommends optimal settings
- Applies enhancements automatically
- Generates detailed report

**Best for:**
- Batch processing
- Unknown photo conditions
- Hands-off restoration
- Professional results

**Advantages:**
- ✅ No configuration needed
- ✅ Consistent results
- ✅ Detailed reports
- ✅ Fast processing

### ⚙️ Manual Mode

**What it does:**
- You control all settings
- Customize each parameter
- Fine-tune to your needs

**Best for:**
- Specific requirements
- Artistic control
- Testing settings
- Learning

**Advantages:**
- ✅ Full control
- ✅ Flexible
- ✅ Educational
- ✅ Custom results

## Setting Reference

### Geometric Settings

- **Enable**: Turn on/off
- **Angle Threshold**: Minimum rotation to fix

### Filtering Settings

- **Method**:
  - `bilateral`: Recommended (balance quality/speed)
  - `nlm`: Best quality (slower)
  - `median`: Salt-and-pepper noise
  - `gaussian`: Simple blur (fastest)

- **Strength**: 0.5 (gentle) - 2.0 (aggressive)
- **Combined**: Use bilateral + NLM

### Histogram Settings

- **Method**:
  - `clahe`: Recommended (adaptive)
  - `standard`: Standard equalization
  - `multiscale`: Multi-scale approach

- **Clip Limit**: 1.0 (gentle) - 4.0 (aggressive)
- **Color Balance**: Restore faded colors
- **Local Contrast**: Enhance local details

## Batch Processing

### Upload Multiple Files

1. Click upload area
2. Hold Ctrl (Cmd on Mac)
3. Select multiple photos
4. Or drag-drop multiple files

### Processing

- System processes all automatically
- Shows progress bar
- Displays results in order

### Download

- Download individual files
- Or download all as ZIP

## Results Tab

### View Results

- See before-after comparison
- View detailed analysis (Smart mode)
- Check file sizes
- Download options

### Analysis Report (Smart Mode)

Shows:
- Image metrics (brightness, noise, etc.)
- Quality assessment
- Applied enhancements
- Reason for each step
- Summary notes

## Tips & Tricks

### For Best Results

1. **Use Smart Mode** for batch processing
2. **Check Analysis Report** to understand what was done
3. **Compare before-after** to verify quality
4. **Adjust clip_limit** if too bright/dark (2.0-2.5 recommended)
5. **Use bilateral filter** for balance

### Batch Processing

1. Upload 5-10 photos at once
2. Use Smart Mode (auto-optimized)
3. Process all together
4. Download results

### Custom Enhancement

1. Switch to Manual Mode
2. Adjust parameters
3. Test on 1 photo first
4. Apply to batch if satisfied

## Troubleshooting

### Result is too bright/dark

**Smart Mode:**
- System auto-optimized, usually correct

**Manual Mode:**
- Adjust `Clip Limit` (lower = darker, higher = brighter)
- Try different method (standard vs multiscale)

### Result is still blurry

- Blur cannot be fixed (inherent to image)
- Try `nlm` filter for slight improvement
- Check Analysis Report

### Processing is slow

1. Reduce image size in config
2. Use `bilateral` filter (faster)
3. Skip combined filters
4. Process smaller batches

### Downloaded file is large

- Reduce quality setting (affects file size, not visual)
- Use PNG instead of JPG (or vice versa)

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+B` | Open sidebar |
| `Ctrl+M` | Clear cache |
| `Ctrl+C` | Copy output |

## Next Steps

- See [EXAMPLES.md](EXAMPLES.md) for code examples
- Check [API.md](API.md) for detailed reference