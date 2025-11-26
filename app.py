"""
OLD PHOTO ENHANCEMENT - ENHANCED STREAMLIT WEB APP
Professional, Interactive, Feature-Rich UI

Features:
- 🤖 Smart Auto-Detection
- 📊 Detailed Analytics
- 🎨 Beautiful UI/UX
- 📥 Batch Upload
- 📤 Multiple Export Formats
- 📈 Performance Metrics
- 💾 Session Management

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

# Import modules
from modules.smart_enhancer import SmartEnhancer
from modules.image_analyzer import ImageAnalyzer
from modules.geometric import GeometricCorrection
from modules.filtering import NoiseFiltering
from modules.histogram import HistogramEqualization
from utils.config_manager import ConfigManager
from utils.logger import setup_logger

# ============================================================================
# CONFIG & SETUP
# ============================================================================

st.set_page_config(
    page_title="Old Photo Enhancement | AI Restoration",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

logger = setup_logger(__name__)
config = ConfigManager.load_config('config/settings.json')

# ============================================================================
# CUSTOM CSS - PROFESSIONAL STYLING
# ============================================================================

st.markdown("""
    <style>
    /* Main Container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Headers */
    .header-main {
        text-align: center;
        padding: 40px 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    
    .header-main h1 {
        font-size: 3em;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .header-main p {
        font-size: 1.2em;
        margin: 10px 0 0 0;
        opacity: 0.95;
    }
    
    /* Info Boxes */
    .info-box {
        background: rgba(102, 126, 234, 0.1);
        border: 2px solid #667eea;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    .success-box {
        background: rgba(40, 167, 69, 0.1);
        border: 2px solid #28a745;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    .warning-box {
        background: rgba(255, 193, 7, 0.1);
        border: 2px solid #ffc107;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    .error-box {
        background: rgba(220, 53, 69, 0.1);
        border: 2px solid #dc3545;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        margin: 10px 0;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 25px;
        font-weight: bold;
        transition: transform 0.2s;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: rgba(102, 126, 234, 0.2);
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================

if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'processing_results' not in st.session_state:
    st.session_state.processing_results = {}
if 'total_processed' not in st.session_state:
    st.session_state.total_processed = 0
if 'total_time' not in st.session_state:
    st.session_state.total_time = 0

# ============================================================================
# HEADER
# ============================================================================

st.markdown("""
    <div class="header-main">
        <h1>🖼️ Old Photo Enhancement</h1>
        <p>🤖 AI-Powered Photo Restoration | Bring Your Memories Back to Life</p>
    </div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - NAVIGATION & SETTINGS
# ============================================================================

st.sidebar.title("⚙️ Settings & Navigation")

# Mode Selection
st.sidebar.markdown("### 🎯 Enhancement Mode")
enhancement_mode = st.sidebar.radio(
    "Choose your mode:",
    options=['🤖 Smart (AI Auto-Detect)', '⚙️ Manual (Custom Settings)'],
    help="Smart: System auto-optimizes | Manual: You control everything"
)

# Smart Settings
if enhancement_mode == '🤖 Smart (AI Auto-Detect)':
    st.sidebar.markdown("### 🤖 Smart Enhancement Settings")
    with st.sidebar.expander("Smart Configuration", expanded=True):
        smart_generate_report = st.checkbox("Generate Detailed Report", value=True)
        smart_save_analysis = st.checkbox("Save Analysis Data", value=True)
        st.info("💡 Smart mode automatically analyzes each photo and applies optimal settings!")

# Manual Settings
else:
    st.sidebar.markdown("### ⚙️ Manual Enhancement Settings")
    
    with st.sidebar.expander("🔧 Geometric Correction", expanded=False):
        enable_geometric = st.checkbox("Enable Geometric Correction", value=True)
        angle_threshold = st.slider("Rotation Threshold", 1, 45, 5)
    
    with st.sidebar.expander("🧹 Noise Filtering", expanded=True):
        filter_method = st.selectbox(
            "Filter Method",
            ['bilateral', 'nlm', 'median', 'gaussian'],
            help="bilateral ⭐ recommended - balance quality & speed"
        )
        filter_strength = st.slider("Strength", 0.5, 2.0, 1.0, 0.1)
        enable_combined = st.checkbox("Combined Filters (bilateral + NLM)", value=False)
    
    with st.sidebar.expander("💡 Histogram Enhancement", expanded=True):
        histogram_method = st.selectbox(
            "Enhancement Method",
            ['clahe', 'standard', 'multiscale'],
            help="clahe ⭐ recommended"
        )
        clip_limit = st.slider("Contrast Clip Limit", 1.0, 4.0, 2.0, 0.1)
        enable_color_balance = st.checkbox("Color Balance", value=True)

# Output Settings
st.sidebar.markdown("### 📤 Output Settings")
output_quality = st.sidebar.slider("Output Quality", 70, 100, 95)

# About Section
st.sidebar.markdown("---")
with st.sidebar.expander("ℹ️ About This App"):
    st.markdown("""
    **Old Photo Enhancement v1.0**
    
    🤖 AI-powered restoration system
    
    **Features:**
    - Smart auto-detection
    - Batch processing
    - Detailed reports
    - Multiple formats
    
    **Authors:** Photo Enhancement Team
    
    **License:** MIT
    """)

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
        st.markdown("""
            <div class="info-box">
            <h4>🎯 Supported Formats</h4>
            JPG • PNG • BMP • TIFF<br><br>
            <strong>Max File Size:</strong> 200 MB per file<br>
            <strong>Max Photos:</strong> Unlimited
            </div>
        """, unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Drag and drop your photos here or click to browse",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("""
            <div class="success-box">
            <h4>💡 Pro Tips</h4>
            ✓ Upload multiple files<br>
            ✓ Batch processing<br>
            ✓ Auto-optimized settings<br>
            ✓ Instant download
            </div>
        """, unsafe_allow_html=True)
    
    # File Statistics
    if uploaded_files:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Files Uploaded", len(uploaded_files))
        with col2:
            total_size = sum(f.size for f in uploaded_files) / (1024*1024)
            st.metric("Total Size", f"{total_size:.2f} MB")
        with col3:
            st.metric("Ready to Process", "✅ Yes" if uploaded_files else "❌ No")
    
    st.markdown("---")
    
    # Processing Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        process_button = st.button(
            "🚀 START PROCESSING",
            use_container_width=True,
            type="primary",
            key="process_btn"
        )
    
    # Processing Logic
    if process_button and uploaded_files:
        start_time = time.time()
        
        # Initialize enhancers
        if enhancement_mode == '🤖 Smart (AI Auto-Detect)':
            enhancer = SmartEnhancer()
            use_smart = True
        else:
            use_smart = False
        
        progress_bar = st.progress(0)
        status_container = st.container()
        metrics_container = st.container()
        
        results = {}
        errors = []
        
        with status_container:
            status_text = st.empty()
            details_text = st.empty()
        
        for idx, uploaded_file in enumerate(uploaded_files):
            file_name = uploaded_file.name
            progress = idx / len(uploaded_files)
            progress_bar.progress(min(progress, 0.99))
            
            status_text.markdown(f"<div class='info-box'><strong>Processing:</strong> {file_name} ({idx+1}/{len(uploaded_files)})</div>", unsafe_allow_html=True)
            details_text.text(f"File size: {uploaded_file.size / (1024*1024):.2f} MB")
            
            try:
                # Read file
                image_bytes = uploaded_file.read()
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is None:
                    errors.append(f"❌ {file_name}: Failed to read image")
                    continue
                
                # Process
                if use_smart:
                    result = enhancer.enhance(image)
                    results[file_name] = {
                        'original': result['original'],
                        'enhanced': result['enhanced'],
                        'analysis': result['analysis'],
                        'report': result['report'],
                        'steps_applied': result['steps_applied'],
                        'summary': result['summary'],
                        'is_smart': True
                    }
                else:
                    original = image.copy()
                    
                    if enable_geometric:
                        geo_result = GeometricCorrection().correct(image)
                        image = geo_result['image']
                    
                    if enable_combined:
                        image = NoiseFiltering().combined_filter(image, 'bilateral', 'nlm', filter_strength)
                    else:
                        image = NoiseFiltering().apply_filter(image, filter_method, filter_strength)['image']
                    
                    image = HistogramEqualization().enhance(image, histogram_method, clip_limit)['image']
                    
                    if enable_color_balance:
                        image = HistogramEqualization().color_balance(image)
                    
                    results[file_name] = {
                        'original': original,
                        'enhanced': image,
                        'is_smart': False
                    }
                
            except Exception as e:
                errors.append(f"❌ {file_name}: {str(e)}")
                logger.error(f"Error processing {file_name}: {e}")
        
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
            st.metric("📊 Success Rate", f"{len(results)/(len(results)+len(errors))*100:.0f}%")
        
        if len(results) > 0:
            st.markdown(f"""
                <div class="success-box">
                <h4>✅ Successfully Processed {len(results)} files!</h4>
                Check the <strong>Results</strong> tab to view and download.
                </div>
            """, unsafe_allow_html=True)
        
        if errors:
            st.markdown(f"""
                <div class="warning-box">
                <h4>⚠️ {len(errors)} Error(s) Occurred</h4>
                {'<br>'.join(errors)}
                </div>
            """, unsafe_allow_html=True)
    
    elif process_button and not uploaded_files:
        st.markdown("""
            <div class="error-box">
            <h4>❌ No files uploaded</h4>
            Please upload at least 1 photo before processing.
            </div>
        """, unsafe_allow_html=True)

# ============================================================================
# TAB 2: RESULTS & DOWNLOADS
# ============================================================================

with tab2:
    st.markdown("<h2>📊 Processing Results</h2>", unsafe_allow_html=True)
    
    if st.session_state.processing_results:
        
        # Display stats
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Photos", len(st.session_state.processing_results))
        with col2:
            total_size = sum(r['enhanced'].nbytes for r in st.session_state.processing_results.values()) / (1024*1024)
            st.metric("Total Output Size", f"{total_size:.2f} MB")
        
        st.markdown("---")
        
        # Results
        for idx, (file_name, result) in enumerate(st.session_state.processing_results.items(), 1):
            with st.expander(f"📸 {idx}. {file_name}", expanded=False):
                
                # Analysis Report
                if result.get('is_smart'):
                    st.markdown("### 📊 Analysis Report")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Brightness", f"{result['report']['metrics']['brightness']:.0f}/255")
                        st.metric("Sharpness", f"{result['report']['metrics']['sharpness']:.1%}")
                    with col2:
                        st.metric("Contrast", f"{result['report']['metrics']['contrast']:.1f}")
                        st.metric("Saturation", f"{result['report']['metrics']['color_saturation']:.1%}")
                    with col3:
                        st.metric("Rotation", f"{result['report']['metrics']['rotation_angle']:.1f}°")
                        st.metric("Noise", f"{result['report']['metrics']['noise_level']:.1%}")
                    
                    # Enhancements Applied
                    st.markdown("### ✅ Enhancements Applied")
                    for step in result['steps_applied']:
                        with st.expander(f"📌 {step['step']}"):
                            st.write(f"**Method:** {step['method']}")
                            st.write(f"**Reason:** {step['reason']}")
                            for key, value in step.items():
                                if key not in ['step', 'method', 'reason']:
                                    st.write(f"**{key}:** {value}")
                    
                    st.markdown(f"**Summary:** {result['summary']}")
                
                # Image Comparison
                st.markdown("### 🖼️ Before & After")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Original**")
                    orig_rgb = cv2.cvtColor(result['original'], cv2.COLOR_BGR2RGB)
                    st.image(orig_rgb, use_column_width=True)
                
                with col2:
                    st.markdown("**Enhanced**")
                    enh_rgb = cv2.cvtColor(result['enhanced'], cv2.COLOR_BGR2RGB)
                    st.image(enh_rgb, use_column_width=True)
                
                # Download
                st.markdown("### 📥 Download")
                col1, col2 = st.columns(2)
                
                with col1:
                    is_success, buffer = cv2.imencode('.jpg', result['enhanced'], [cv2.IMWRITE_JPEG_QUALITY, output_quality])
                    st.download_button(
                        label=f"⬇️ JPG ({len(buffer.tobytes())/(1024):.0f}KB)",
                        data=buffer.tobytes(),
                        file_name=f"enhanced_{Path(file_name).stem}.jpg",
                        mime="image/jpeg",
                        use_container_width=True
                    )
                
                with col2:
                    is_success, buffer = cv2.imencode('.png', result['enhanced'])
                    st.download_button(
                        label=f"⬇️ PNG ({len(buffer.tobytes())/(1024):.0f}KB)",
                        data=buffer.tobytes(),
                        file_name=f"enhanced_{Path(file_name).stem}.png",
                        mime="image/png",
                        use_container_width=True
                    )
    
    else:
        st.info("📌 Process photos in the **Upload & Process** tab first!")

# ============================================================================
# TAB 3: ANALYTICS
# ============================================================================

with tab3:
    st.markdown("<h2>📈 Analytics & Performance</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Processed", st.session_state.total_processed)
    with col2:
        st.metric("Total Time", f"{st.session_state.total_time:.1f}s")
    with col3:
        if st.session_state.total_processed > 0:
            avg_time = st.session_state.total_time / st.session_state.total_processed
            st.metric("Avg Time/Photo", f"{avg_time:.1f}s")
    with col4:
        st.metric("Mode", "Smart 🤖" if enhancement_mode == '🤖 Smart (AI Auto-Detect)' else "Manual ⚙️")
    
    st.markdown("---")
    
    if st.session_state.processing_results:
        st.markdown("### 📊 Result Statistics")
        
        # Analysis if smart mode
        if any(r.get('is_smart') for r in st.session_state.processing_results.values()):
            st.markdown("#### 🤖 Smart Mode Analysis")
            
            # Collect metrics
            metrics_data = {
                'Brightness': [],
                'Noise': [],
                'Contrast': [],
                'Saturation': [],
            }
            
            for result in st.session_state.processing_results.values():
                if result.get('is_smart'):
                    m = result['report']['metrics']
                    metrics_data['Brightness'].append(m['brightness'])
                    metrics_data['Noise'].append(m['noise_level'])
                    metrics_data['Contrast'].append(m['contrast'])
                    metrics_data['Saturation'].append(m['color_saturation'])
            
            # Display stats
            for metric_name, values in metrics_data.items():
                if values:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(f"{metric_name} - Avg", f"{np.mean(values):.2f}")
                    with col2:
                        st.metric(f"{metric_name} - Min", f"{np.min(values):.2f}")
                    with col3:
                        st.metric(f"{metric_name} - Max", f"{np.max(values):.2f}")

# ============================================================================
# TAB 4: GUIDE
# ============================================================================

with tab4:
    st.markdown("<h2>📚 User Guide</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Getting Started
        
        1. **Upload Photos**
           - Click upload area or drag-drop
           - Support JPG, PNG, BMP, TIFF
        
        2. **Choose Mode**
           - 🤖 Smart: AI auto-optimizes
           - ⚙️ Manual: You control settings
        
        3. **Process**
           - Click "START PROCESSING"
           - Wait for completion
        
        4. **Download**
           - View results in Results tab
           - Download JPG or PNG
        """)
    
    with col2:
        st.markdown("""
        ### ⚙️ Smart Mode Features
        
        **Auto-Detection:**
        - Rotation angle
        - Noise level
        - Brightness
        - Contrast
        - Color fading
        
        **Auto-Optimization:**
        - Optimal filter selection
        - Parameter auto-calculation
        - Detailed reporting
        """)
    
    st.markdown("---")
    st.markdown("""
    ### 💡 Tips & Tricks
    
    - **Batch Processing**: Upload multiple photos at once
    - **Smart Mode**: Best for unknown or varied photos
    - **Manual Mode**: For specific customization
    - **Reports**: Check detailed analysis in Results tab
    - **Export**: Download as JPG or PNG
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; opacity: 0.7;">
    <p>
        🖼️ <strong>Old Photo Enhancement v1.0</strong> | 
        🤖 AI-Powered Restoration | 
        Made with ❤️
    </p>
</div>
""", unsafe_allow_html=True)