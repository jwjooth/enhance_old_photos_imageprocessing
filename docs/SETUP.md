# 🚀 Setup Guide - Old Photo Enhancement

## Prerequisites

- Python 3.11 or higher
- pip (Python package manager)
- Git (optional, for cloning)
- 2GB free disk space

## Installation Steps

### 1. Clone Repository

\`\`\`bash
git clone https://github.com/yourusername/old-photo-enhancement.git
cd old-photo-enhancement
\`\`\`

### 2. Create Virtual Environment

**Windows:**
\`\`\`bash
python -m venv venv
venv\Scripts\activate
\`\`\`

**macOS/Linux:**
\`\`\`bash
python3 -m venv venv
source venv/bin/activate
\`\`\`

You should see `(venv)` in your terminal prompt.

### 3. Install Dependencies

\`\`\`bash
pip install -r requirements.txt
\`\`\`

Verify installation:
\`\`\`bash
python -c "import cv2, streamlit, numpy; print('✅ All installed!')"
\`\`\`

### 4. Create Directories

\`\`\`bash
mkdir -p input output logs .streamlit
\`\`\`

### 5. Verify Setup

\`\`\`bash
# Test modules
python -c "from modules.smart_enhancer import SmartEnhancer; print('✅ Setup OK')"

# Run tests
pytest
\`\`\`

## Running the Application

### Streamlit Web App (Recommended)

\`\`\`bash
streamlit run app.py
\`\`\`

Opens: http://localhost:8501

### CLI Version

\`\`\`bash
python main.py
\`\`\`

### Docker

\`\`\`bash
docker-compose up
\`\`\`

Opens: http://localhost:8501

## Configuration

Edit `config/settings.json` to customize:
- Thresholds
- Parameters
- Export options

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'cv2'"

\`\`\`bash
pip install opencv-python
\`\`\`

### Issue: "Address already in use"

\`\`\`bash
streamlit run app.py --server.port 8502
\`\`\`

### Issue: Virtual environment not activating

**Windows:**
\`\`\`bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
venv\Scripts\activate
\`\`\`

### Issue: Permission denied

\`\`\`bash
chmod +x venv/bin/activate
source venv/bin/activate
\`\`\`

## Next Steps

- Read [USAGE.md](USAGE.md) for how to use
- Check [EXAMPLES.md](EXAMPLES.md) for code examples
- See [API.md](API.md) for detailed API reference