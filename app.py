"""
OLD PHOTO ENHANCEMENT - ULTIMATE PROFESSIONAL STREAMLIT APP
The BEST photo restoration experience with AI + Advanced Features

🤖 Smart Auto-Detection + 🎨 Advanced Processing + ⚙️ Manual Control

Run: streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import json
import time
from pathlib import Path
from datetime import datetime
import logging

# Import all modules
from modules.smart_enhancer import SmartEnhancer
from modules.image_analyzer import ImageAnalyzer
from modules.geometric import GeometricCorrection
from modules.filtering import NoiseFiltering
from modules.histogram import HistogramEqualization
from modules.processors import (
    rotate_image,
    apply_zoom,
    adjust_brightness_contrast,
    apply_denoise,
    apply_enhancement,
    apply_sharpening,
    apply_color_tone,
    detect_all_faces,
    create_multi_face_mask,
    master_pipeline,
)
from modules.auto_tuner import analyze_and_recommend, apply_preset
from utils.config_manager import ConfigManager
from utils.logger import setup_logger

# ============================================================================
# CONFIG & SETUP
# ============================================================================

st.set_page_config(
    page_title="Old Photo Enhancement | AI Restoration",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

logger = setup_logger(__name__)
config = ConfigManager.load_config("config/settings.json")

# ============================================================================
# CUSTOM CSS - PREMIUM STYLING
# ============================================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .header-main {
        text-align: center;
        padding: 50px 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        animation: fadeIn 0.5s ease-in;
    }
    
    .header-main h1 {
        font-size: 3.5em;
        margin: 0;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.4);
        font-weight: 700;
    }
    
    .header-main p {
        font-size: 1.3em;
        margin: 15px 0 0 0;
        opacity: 0.95;
        font-weight: 600;
    }
    
    .feature-box {
        background: rgba(102, 126, 234, 0.1);
        border: 2px solid #667eea;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        transition: all 0.3s ease;
    }
    
    .feature-box:hover {
        background: rgba(102, 126, 234, 0.2);
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.2);
    }
    
    .success-box {
        background: rgba(40, 167, 69, 0.1);
        border: 2px solid #28a745;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
    }
    
    .warning-box {
        background: rgba(255, 193, 7, 0.1);
        border: 2px solid #ffc107;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
    }
    
    .error-box {
        background: rgba(220, 53, 69, 0.1);
        border: 2px solid #dc3545;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 30px !important;
        font-weight: 700 !important;
        transition: all 0.3s ease !important;
        font-size: 1.1em !important;
    }
    
    .stButton>button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4) !important;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 2px solid rgba(102, 126, 234, 0.3);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "processing_results" not in st.session_state:
    st.session_state.processing_results = {}
if "total_processed" not in st.session_state:
    st.session_state.total_processed = 0
if "total_time" not in st.session_state:
    st.session_state.total_time = 0
if "auto_recommendations" not in st.session_state:
    st.session_state.auto_recommendations = {}

# ============================================================================
# HEADER
# ============================================================================

st.markdown(
    """
    <div class="header-main">
        <h1>🖼️ Old Photo Enhancement</h1>
        <p>🤖 AI-Powered Professional Photo Restoration</p>
        <p style="font-size: 0.9em; margin-top: 10px; opacity: 0.8;">Restore Your Precious Memories with Advanced AI Technology</p>
    </div>
""",
    unsafe_allow_html=True,
)

# ============================================================================
# SIDEBAR - NAVIGATION & SETTINGS
# ============================================================================

st.sidebar.title("⚙️ Settings & Navigation")

# Mode Selection
st.sidebar.markdown("### 🎯 Enhancement Mode")
enhancement_mode = st.sidebar.radio(
    "Choose your enhancement approach:",
    options=["🤖 Smart (AI Auto-Detect)", "🎨 Advanced (Processors)", "⚙️ Manual (Custom Settings)"],
    help="Smart: Full automation | Advanced: Precise control with AI | Manual: Complete control",
)

# ============================================================================
# MODE 1: SMART ENHANCEMENT
# ============================================================================

if enhancement_mode == "🤖 Smart (AI Auto-Detect)":
    st.sidebar.markdown("### 🤖 Smart Enhancement")
    with st.sidebar.expander("⚙️ Configuration", expanded=True):
        smart_generate_report = st.checkbox("📊 Generate Detailed Report", value=True)
        smart_save_analysis = st.checkbox("💾 Save Analysis Data", value=True)
        st.success("💡 Full automatic mode - No configuration needed!")

# ============================================================================
# MODE 2: ADVANCED PROCESSORS
# ============================================================================

elif enhancement_mode == "🎨 Advanced (Processors)":
    st.sidebar.markdown("### 🎨 Advanced Processing Engine")

    with st.sidebar.expander("🎯 Quick Presets", expanded=True):
        preset_mode = st.selectbox(
            "Select Enhancement Preset",
            ["Auto Detect", "Portrait Mode", "Old Photo Cleanup", "Custom"],
        )

        if preset_mode != "Custom":
            st.info(f"✅ Using '{preset_mode}' preset")

    if preset_mode == "Custom":
        with st.sidebar.expander("🔧 Geometry", expanded=False):
            rotation = st.slider("Rotation (degrees)", -45, 45, 0)
            zoom = st.slider("Zoom Factor", 1.0, 3.0, 1.0, 0.1)
            brightness = st.slider("Brightness", -100, 100, 0)
            contrast = st.slider("Contrast", -100, 100, 0)

        with st.sidebar.expander("🧹 Denoising", expanded=True):
            bg_denoise_algo = st.selectbox(
                "Background Denoise", ["None", "Gaussian Blur", "Median Blur", "Bilateral Filter"]
            )
            bg_denoise_val = st.slider("BG Denoise Strength", 0, 10, 3)

            face_denoise_algo = st.selectbox(
                "Face Denoise",
                ["None", "Gaussian Blur", "Median Blur", "Bilateral Filter", "NLM (Premium)"],
            )
            face_denoise_val = st.slider("Face Denoise Strength", 0, 10, 5)

        with st.sidebar.expander("💡 Enhancement", expanded=False):
            face_enhance_algo = st.selectbox(
                "Enhancement Method", ["None", "Histogram Eq", "CLAHE", "Gamma Correction"]
            )
            face_enhance_val = st.slider("Enhancement Level", 0, 100, 50)

        with st.sidebar.expander("✨ Sharpening", expanded=False):
            face_sharpen_algo = st.selectbox(
                "Sharpening Method", ["None", "Unsharp Masking", "Laplacian", "High Pass Overlay"]
            )
            face_sharpen_val = st.slider("Sharpening Amount", 0, 100, 30)

        with st.sidebar.expander("🎨 Color & Tone", expanded=False):
            color_tone = st.selectbox(
                "Color Tone", ["B&W (Default)", "Sepia (Vintage)", "Selenium (Cool)"]
            )
            enable_scratch = st.checkbox("Enable Scratch Removal", False)
            if enable_scratch:
                scratch_thresh = st.slider("Scratch Threshold", 5, 50, 20)

# ============================================================================
# MODE 3: MANUAL ENHANCEMENT
# ============================================================================

else:  # Manual Mode
    st.sidebar.markdown("### ⚙️ Manual Enhancement")

    with st.sidebar.expander("🔧 Geometric Correction", expanded=False):
        enable_geometric = st.checkbox("🔄 Enable Geometric Correction", value=True)
        angle_threshold = st.slider("Rotation Threshold", 1, 45, 5)

    with st.sidebar.expander("🧹 Noise Filtering", expanded=True):
        filter_method = st.selectbox(
            "Filter Method",
            ["bilateral", "nlm", "median", "gaussian"],
            help="bilateral ⭐ recommended",
        )
        filter_strength = st.slider("Filter Strength", 0.5, 2.0, 1.0, 0.1)
        enable_combined = st.checkbox("🔗 Combined Filters", value=False)

    with st.sidebar.expander("💡 Histogram Enhancement", expanded=True):
        histogram_method = st.selectbox(
            "Enhancement Method", ["clahe", "standard", "multiscale"], help="clahe ⭐ recommended"
        )
        clip_limit = st.slider("Contrast Clip Limit", 1.0, 4.0, 2.0, 0.1)
        enable_color_balance = st.checkbox("🎨 Color Balance", value=True)

# Output Settings (untuk semua mode)
st.sidebar.markdown("### 📤 Output Settings")
output_quality = st.sidebar.slider("Output Quality", 70, 100, 95)

# About Section
st.sidebar.markdown("---")
with st.sidebar.expander("ℹ️ About This App"):
    st.markdown(
        """
    **Old Photo Enhancement v2.0**
    
    🤖 Next-generation AI-powered restoration
    
    **Modes:**
    - 🤖 Smart: Full automation
    - 🎨 Advanced: Precise control
    - ⚙️ Manual: Traditional control
    
    **Features:**
    - Batch processing
    - Detailed analytics
    - Multiple export formats
    - Professional quality
    
    **License:** 
    - Njo Darren Gavriel Valkalino S
    - Johan Julius Rumahorbo
    - Jordan Theovandy
    """
    )

# ============================================================================
# MAIN CONTENT - TABS
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Process", "📊 Results", "📈 Analytics", "📚 Guide"])

# ============================================================================
# TAB 1: UPLOAD & PROCESS
# ============================================================================

with tab1:
    st.markdown("<h2>📥 Upload Your Photos</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown(
            """
            <div class="feature-box">
            <h4>🎯 Supported Formats</h4>
            JPG • PNG • BMP • TIFF<br><br>
            <strong>Max File Size:</strong> 200 MB per file
            </div>
        """,
            unsafe_allow_html=True,
        )

        uploaded_files = st.file_uploader(
            "Drag and drop your photos or click to browse",
            type=["jpg", "jpeg", "png", "bmp", "tiff"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

    with col2:
        st.markdown(
            """
            <div class="success-box">
            <h4>💡 Pro Tips</h4>
            ✓ Multiple files<br>
            ✓ Batch processing<br>
            ✓ Auto-optimization<br>
            ✓ Instant download
            </div>
        """,
            unsafe_allow_html=True,
        )

    # File Statistics
    if uploaded_files:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📁 Files", len(uploaded_files))
        with col2:
            total_size = sum(f.size for f in uploaded_files) / (1024 * 1024)
            st.metric("💾 Total Size", f"{total_size:.2f} MB")
        with col3:
            st.metric("✅ Status", "Ready")

    st.markdown("---")

    # Processing Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        process_button = st.button(
            "🚀 START PROCESSING", use_container_width=True, type="primary", key="process_btn"
        )

    # ============================================================================
    # PROCESSING LOGIC
    # ============================================================================

    if process_button and uploaded_files:
        start_time = time.time()

        # Initialize modules
        if enhancement_mode == "🤖 Smart (AI Auto-Detect)":
            enhancer = SmartEnhancer()
            use_smart = True
            use_advanced = False
        elif enhancement_mode == "🎨 Advanced (Processors)":
            use_smart = False
            use_advanced = True
        else:
            use_smart = False
            use_advanced = False

        progress_bar = st.progress(0)
        status_container = st.container()

        results = {}
        errors = []

        with status_container:
            status_text = st.empty()
            details_text = st.empty()

        for idx, uploaded_file in enumerate(uploaded_files):
            file_name = uploaded_file.name
            progress = idx / len(uploaded_files)
            progress_bar.progress(min(progress, 0.99))

            status_text.markdown(
                f"<div class='feature-box'><strong>Processing:</strong> {file_name} ({idx+1}/{len(uploaded_files)})</div>",
                unsafe_allow_html=True,
            )
            details_text.text(f"Size: {uploaded_file.size / (1024*1024):.2f} MB")

            try:
                # Read file
                image_bytes = uploaded_file.read()
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if image is None:
                    errors.append(f"❌ {file_name}: Failed to read")
                    continue

                # ===== SMART MODE =====
                if use_smart:
                    result = enhancer.enhance(image)
                    results[file_name] = {
                        "original": result["original"],
                        "enhanced": result["enhanced"],
                        "analysis": result["analysis"],
                        "report": result["report"],
                        "steps_applied": result["steps_applied"],
                        "summary": result["summary"],
                        "is_smart": True,
                    }

                # ===== ADVANCED MODE =====
                elif use_advanced:
                    if preset_mode == "Auto Detect":
                        recommendations = analyze_and_recommend(image)
                        config_advanced = apply_preset("Old Photo Cleanup")
                        config_advanced.update(recommendations)
                    elif preset_mode == "Custom":
                        config_advanced = {
                            "rotation": rotation,
                            "zoom": zoom,
                            "brightness": brightness,
                            "contrast": contrast,
                            "bg_denoise_algo": bg_denoise_algo,
                            "bg_denoise_val": bg_denoise_val,
                            "face_denoise_algo": face_denoise_algo,
                            "face_denoise_val": face_denoise_val,
                            "face_enhance_algo": face_enhance_algo,
                            "face_enhance_val": face_enhance_val,
                            "face_sharpen_algo": face_sharpen_algo,
                            "face_sharpen_val": face_sharpen_val,
                            "color_tone": color_tone,
                            "enable_scratch": enable_scratch,
                            "scratch_thresh": scratch_thresh if enable_scratch else 20,
                            "roi_feather": 51,
                        }
                    else:
                        config_advanced = apply_preset(preset_mode)

                    enhanced, roi_mask, scratch_mask = master_pipeline(image, config_advanced)
                    results[file_name] = {
                        "original": image,
                        "enhanced": enhanced,
                        "is_smart": False,
                        "mode": "advanced",
                    }

                # ===== MANUAL MODE =====
                else:
                    original = image.copy()

                    if enable_geometric:
                        image = GeometricCorrection().correct(image)["image"]

                    if enable_combined:
                        image = NoiseFiltering().combined_filter(
                            image, "bilateral", "nlm", filter_strength
                        )
                    else:
                        image = NoiseFiltering().apply_filter(
                            image, filter_method, filter_strength
                        )["image"]

                    image = HistogramEqualization().enhance(image, histogram_method, clip_limit)[
                        "image"
                    ]

                    if enable_color_balance:
                        image = HistogramEqualization().color_balance(image)

                    results[file_name] = {
                        "original": original,
                        "enhanced": image,
                        "is_smart": False,
                    }

            except Exception as e:
                errors.append(f"❌ {file_name}: {str(e)}")
                logger.error(f"Error: {e}")

        progress_bar.progress(1.0)

        # Results Summary
        st.session_state.processing_results = results
        st.session_state.total_processed += len(results)
        st.session_state.total_time += time.time() - start_time

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("✅ Processed", len(results))
        with col2:
            st.metric("⚠️ Errors", len(errors))
        with col3:
            st.metric("⏱️ Time", f"{time.time() - start_time:.1f}s")
        with col4:
            st.metric(
                "📊 Success",
                f"{len(results)/(len(results)+len(errors))*100 if (len(results)+len(errors))>0 else 0:.0f}%",
            )

        if len(results) > 0:
            st.markdown(
                f"""
                <div class="success-box">
                <h4>✅ Successfully Processed {len(results)} files!</h4>
                Check the <strong>Results</strong> tab to view and download.
                </div>
            """,
                unsafe_allow_html=True,
            )

        if errors:
            st.markdown(
                f"""
                <div class="warning-box">
                <h4>⚠️ {len(errors)} Error(s)</h4>
                {'<br>'.join(errors)}
                </div>
            """,
                unsafe_allow_html=True,
            )

    elif process_button and not uploaded_files:
        st.markdown(
            """
            <div class="error-box">
            <h4>❌ No files uploaded</h4>
            Please upload at least 1 photo.
            </div>
        """,
            unsafe_allow_html=True,
        )

# ============================================================================
# TAB 2: RESULTS & DOWNLOADS
# ============================================================================

with tab2:
    st.markdown("<h2>📊 Processing Results</h2>", unsafe_allow_html=True)

    if st.session_state.processing_results:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📁 Total Photos", len(st.session_state.processing_results))
        with col2:
            total_size = sum(
                r["enhanced"].nbytes for r in st.session_state.processing_results.values()
            ) / (1024 * 1024)
            st.metric("💾 Output Size", f"{total_size:.2f} MB")

        st.markdown("---")

        for idx, (file_name, result) in enumerate(st.session_state.processing_results.items(), 1):
            with st.expander(f"📸 {idx}. {file_name}", expanded=False):

                if result.get("is_smart"):
                    st.markdown("### 📊 Smart Analysis")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "📉 Brightness", f"{result['report']['metrics']['brightness']:.0f}/255"
                        )
                        st.metric("✨ Sharpness", f"{result['report']['metrics']['sharpness']:.1%}")
                    with col2:
                        st.metric("📊 Contrast", f"{result['report']['metrics']['contrast']:.1f}")
                        st.metric(
                            "🎨 Saturation",
                            f"{result['report']['metrics']['color_saturation']:.1%}",
                        )
                    with col3:
                        st.metric(
                            "🔄 Rotation", f"{result['report']['metrics']['rotation_angle']:.1f}°"
                        )
                        st.metric("📢 Noise", f"{result['report']['metrics']['noise_level']:.1%}")

                    st.markdown("### ✅ Enhancements")
                    for step in result["steps_applied"]:
                        with st.expander(f"📌 {step['step']}"):
                            st.write(f"**Method:** {step['method']}")
                            st.write(f"**Reason:** {step['reason']}")
                    st.info(f"**Summary:** {result['summary']}")

                st.markdown("### 🖼️ Before & After")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Original**")
                    orig_rgb = cv2.cvtColor(result["original"], cv2.COLOR_BGR2RGB)
                    st.image(orig_rgb, use_container_width=True)
                with col2:
                    st.markdown("**Enhanced**")
                    enh_rgb = cv2.cvtColor(result["enhanced"], cv2.COLOR_BGR2RGB)
                    st.image(enh_rgb, use_container_width=True)

                st.markdown("### 📥 Download")
                col1, col2 = st.columns(2)
                with col1:
                    _, buffer = cv2.imencode(
                        ".jpg", result["enhanced"], [cv2.IMWRITE_JPEG_QUALITY, output_quality]
                    )
                    st.download_button(
                        "⬇️ JPG",
                        buffer.tobytes(),
                        f"enhanced_{Path(file_name).stem}.jpg",
                        "image/jpeg",
                        use_container_width=True,
                    )
                with col2:
                    _, buffer = cv2.imencode(".png", result["enhanced"])
                    st.download_button(
                        "⬇️ PNG",
                        buffer.tobytes(),
                        f"enhanced_{Path(file_name).stem}.png",
                        "image/png",
                        use_container_width=True,
                    )
    else:
        st.info("📌 Process photos first in the Upload tab!")

# ============================================================================
# TAB 3: ANALYTICS
# ============================================================================

with tab3:
    st.markdown("<h2>📈 Performance Analytics</h2>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Total", st.session_state.total_processed)
    with col2:
        st.metric("⏱️ Time", f"{st.session_state.total_time:.1f}s")
    with col3:
        if st.session_state.total_processed > 0:
            st.metric(
                "⚡ Avg/Photo",
                f"{st.session_state.total_time / st.session_state.total_processed:.1f}s",
            )
    with col4:
        st.metric("🎯 Mode", enhancement_mode.split("(")[0].strip())

# ============================================================================
# TAB 4: GUIDE
# ============================================================================

with tab4:
    st.markdown("<h2>📚 User Guide</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
        ### 🎯 Getting Started
        
        **Smart Mode:**
        1. Upload photos
        2. Click START
        3. Download
        
        **Advanced Mode:**
        1. Choose preset
        2. Upload photos
        3. Click START
        
        **Manual Mode:**
        1. Adjust settings
        2. Upload photos
        3. Click START
        """
        )

    with col2:
        st.markdown(
            """
        ### ✨ Features
        
        🤖 AI Auto-Detection
        🎨 Advanced Processors
        ⚙️ Manual Control
        📊 Detailed Reports
        📤 Multiple Formats
        ⚡ Batch Processing
        """
        )

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    """
<div style="text-align: center; padding: 20px;">
    <p style="opacity: 0.8;">
        🖼️ <strong>Old Photo Enhancement v2.0</strong> | 
        🤖 AI-Powered Restoration | 
        Made with ❤️ for Preserving Memories
    </p>
</div>
""",
    unsafe_allow_html=True,
)