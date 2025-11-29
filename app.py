import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import zipfile
from streamlit_image_comparison import image_comparison

# Import modules
import processors
import auto_tuner

st.set_page_config(page_title="RetroFix Pro", layout="wide", page_icon="🎞️")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stExpander { border: 1px solid #444; border-radius: 8px; }
    div[data-testid="stMetricValue"] { font-size: 1.2rem; }
    button[kind="secondary"] { border-color: #4CAF50; color: #4CAF50; }
    /* Highlight for Preset Button */
    div.stButton > button:first-child { font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- STRICT PRESET DEFINITIONS ---
# Logic Requirement: Force Update ALL slider values.
PRESET_LIBS = {
    "Full Body (Far / Wide Shot)": {
        'description': "Good for general cleaning without destroying small facial details.",
        'config': {
            'enable_scratch': False,
            'scratch_thresh': 20, # Default safety
            'bg_denoise_algo': 'Bilateral Filter',
            'bg_denoise_val': 18, # Avg of 15-20
            'roi_feather': 50,
            'face_denoise_algo': 'NLM (Premium)',
            'face_denoise_val': 4,
            'face_enhance_algo': 'Gamma Correction',
            'face_enhance_val': 38, # Avg of 35-40
            'face_sharpen_algo': 'High Pass Overlay',
            'face_sharpen_val': 45  # Avg of 35-55
        }
    },
    "Full Face (Close Up / Portrait)": {
        'description': "High detail retention, stronger smoothing on skin but precise edges.",
        'config': {
            'enable_scratch': False,
            'scratch_thresh': 20, # Default safety
            'bg_denoise_algo': 'Bilateral Filter',
            'bg_denoise_val': 18, # Avg of 16-20
            'roi_feather': 95,    # Avg of 89-100
            'face_denoise_algo': 'NLM (Premium)',
            'face_denoise_val': 14,
            'face_enhance_algo': 'CLAHE',
            'face_enhance_val': 50, # Avg of 40-60
            'face_sharpen_algo': 'High Pass Overlay',
            'face_sharpen_val': 13
        }
    }
}

# --- STATE MANAGEMENT INIT ---
if 'image_configs' not in st.session_state:
    st.session_state['image_configs'] = {} # Key: Filename, Value: Config Dict

# --- HELPER: GET/SET CONFIG ---
def get_default_config():
    """Returns a neutral starting configuration."""
    return {
        'enable_scratch': False, 'scratch_thresh': 20,
        'bg_denoise_algo': 'None', 'bg_denoise_val': 0,
        'roi_feather': 50,
        'face_denoise_algo': 'None', 'face_denoise_val': 0,
        'face_enhance_algo': 'None', 'face_enhance_val': 0,
        'face_sharpen_algo': 'None', 'face_sharpen_val': 0
    }

def get_config(filename):
    if filename not in st.session_state['image_configs']:
        st.session_state['image_configs'][filename] = get_default_config()
    return st.session_state['image_configs'][filename]

def update_config(filename, key, value):
    st.session_state['image_configs'][filename][key] = value

def apply_strict_preset(filename, preset_name):
    """
    Overwrites the current image config with Strict Preset values.
    Ensures no residual settings remain.
    """
    if preset_name in PRESET_LIBS:
        target_conf = PRESET_LIBS[preset_name]['config']
        # Force update all keys defined in the preset
        for key, val in target_conf.items():
            st.session_state['image_configs'][filename][key] = val

# --- MAIN UI ---
st.title("🎞️ RetroFix: Auto-Grayscale Restoration")
st.markdown("Automatic B&W Foundation | **Multi-Face Detection** | Smart Scratch Removal")

# 1. FILE UPLOADER
uploaded_files = st.sidebar.file_uploader("Upload Old Photos", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    # Sidebar Selection
    file_map = {f.name: f for f in uploaded_files}
    selected_filename = st.sidebar.selectbox("Select Image to Edit:", list(file_map.keys()))
    active_file = file_map[selected_filename]
    
    # Load Image
    file_bytes = np.asarray(bytearray(active_file.read()), dtype=np.uint8)
    original_loaded = cv2.imdecode(file_bytes, 1)
    
    # --- AUTO GRAYSCALE FOUNDATION ---
    gray_temp = cv2.cvtColor(original_loaded, cv2.COLOR_BGR2GRAY)
    original_image = cv2.cvtColor(gray_temp, cv2.COLOR_GRAY2BGR)
    
    # Load State for this specific image
    cfg = get_config(selected_filename)

    # --- SIDEBAR CONTROLS ---
    with st.sidebar:
        st.info("ℹ️ Image automatically converted to Grayscale.")
        st.header("🎛️ Restoration Controls")
        
        # --- NEW PRESET SYSTEM ---
        st.subheader("⚡ Quick Presets")
        preset_choice = st.selectbox(
            "Select Mode:", 
            list(PRESET_LIBS.keys()), 
            index=0,
            key="preset_selector"
        )
        
        # Display description
        st.caption(PRESET_LIBS[preset_choice]['description'])
        
        # Button to Apply
        if st.button(f"Apply '{preset_choice}' Settings", use_container_width=True):
            apply_strict_preset(selected_filename, preset_choice)
            st.success(f"Loaded: {preset_choice}")
            st.rerun() # Force UI refresh to show new slider values

        st.markdown("---")

        # EXPANDER 1: REPAIR & SCRATCHES (BG Focus)
        with st.expander("🛠️ Repair & Scratches (Background)", expanded=False):
            st.caption("Focus on fixing background noise and scratches.")
            
            enable_scratch = st.toggle("Enable Smart Scratch Removal", value=cfg['enable_scratch'])
            update_config(selected_filename, 'enable_scratch', enable_scratch)
            
            if enable_scratch:
                scratch_thresh = st.slider("Scratch Threshold", 0, 255, cfg['scratch_thresh'], help="Lower = more sensitive")
                update_config(selected_filename, 'scratch_thresh', scratch_thresh)
                show_scratch_mask = st.toggle("Show Red Scratch Mask (Debug)", value=False)
            else:
                show_scratch_mask = False
            
            st.markdown("---")
            
            bg_denoise_algo = st.selectbox(
                "BG Denoise Algo", 
                ["None", "Gaussian Blur", "Median Blur", "Bilateral Filter"], 
                index=["None", "Gaussian Blur", "Median Blur", "Bilateral Filter"].index(cfg['bg_denoise_algo'])
            )
            update_config(selected_filename, 'bg_denoise_algo', bg_denoise_algo)
            
            bg_denoise_val = st.slider("BG Denoise Strength", 0, 30, cfg['bg_denoise_val'])
            update_config(selected_filename, 'bg_denoise_val', bg_denoise_val)

        # EXPANDER 2: FACE / ROI DETAILS
        with st.expander("👤 Face / ROI Details", expanded=True):
            st.caption("Faces are **detected automatically**. Adjust algorithms below.")
            
            roi_feather = st.slider("Mask Feathering", 1, 150, cfg['roi_feather'], step=2)
            update_config(selected_filename, 'roi_feather', roi_feather)
            
            st.markdown("---")
            
            # Face Denoise
            f_denoise = st.selectbox(
                "Face Denoise", 
                ["None", "NLM (Premium)", "Bilateral Filter"], 
                index=["None", "NLM (Premium)", "Bilateral Filter"].index(cfg['face_denoise_algo'])
            )
            update_config(selected_filename, 'face_denoise_algo', f_denoise)
            
            if f_denoise != "None":
                fd_val = st.slider("Face Denoise Level", 0, 30, cfg['face_denoise_val'])
                update_config(selected_filename, 'face_denoise_val', fd_val)

            # Face Enhance
            f_enhance = st.selectbox(
                "Face Enhance", 
                ["None", "CLAHE", "Histogram Eq", "Gamma Correction"], 
                index=["None", "CLAHE", "Histogram Eq", "Gamma Correction"].index(cfg['face_enhance_algo'])
            )
            update_config(selected_filename, 'face_enhance_algo', f_enhance)
            
            if f_enhance != "None":
                fe_val = st.slider("Enhance Level", 0, 100, cfg['face_enhance_val'])
                update_config(selected_filename, 'face_enhance_val', fe_val)
                
            # Face Sharpening
            f_sharpen = st.selectbox(
                "Face Sharpening", 
                ["None", "Unsharp Masking", "Laplacian", "High Pass Overlay"], 
                index=["None", "Unsharp Masking", "Laplacian", "High Pass Overlay"].index(cfg['face_sharpen_algo'])
            )
            update_config(selected_filename, 'face_sharpen_algo', f_sharpen)
            
            if f_sharpen != "None":
                fs_val = st.slider("Sharpen Amount", 0, 100, cfg['face_sharpen_val'])
                update_config(selected_filename, 'face_sharpen_val', fs_val)

    # --- MAIN PROCESSING ---
    try:
        final_img_bgr, roi_mask, scratch_mask_vis = processors.master_pipeline(original_image, cfg)
        
        # Display Setup
        final_img_rgb = cv2.cvtColor(final_img_bgr, cv2.COLOR_BGR2RGB)
        original_img_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # Visual Debug Overlay for Scratch
        if show_scratch_mask and cfg['enable_scratch']:
            red_mask = np.zeros_like(final_img_rgb)
            red_mask[:,:,0] = 255 
            mask_indices = scratch_mask_vis > 0
            final_img_rgb[mask_indices] = cv2.addWeighted(final_img_rgb[mask_indices], 0.7, red_mask[mask_indices], 0.3, 0)

        # 1. Debug Mask Visualization
        with st.expander("👁️ View Masks (ROI & Scratch)", expanded=False):
            dc1, dc2 = st.columns(2)
            dc1.image(roi_mask, caption="Auto-Detected Faces Mask", use_container_width=True, clamp=True)
            if cfg['enable_scratch']:
                dc2.image(scratch_mask_vis, caption="Scratch Detection Mask", use_container_width=True)
            else:
                dc2.info("Scratch Removal Disabled")

        # 2. Juxtapose Viewer
        st.subheader("Comparison View (B&W)")
        image_comparison(
            img1=original_img_rgb,
            img2=final_img_rgb,
            label1="Original (Gray Base)",
            label2="Restored Result",
            width=700,
            starting_position=50,
            show_labels=True,
            make_responsive=True,
            in_memory=True
        )

        # 3. Export System
        st.markdown("---")
        st.header("💾 Export Options")
        
        col_ex1, col_ex2 = st.columns([1, 2])
        
        with col_ex1:
            fmt = st.radio("Format", ["PNG", "JPG"], horizontal=True)
            is_success, buffer = cv2.imencode(f".{fmt.lower()}", final_img_bgr)
            byte_io = io.BytesIO(buffer)
            
            st.download_button(
                label="⬇️ Download This Image",
                data=byte_io,
                file_name=f"restored_{selected_filename.rsplit('.', 1)[0]}.{fmt.lower()}",
                mime=f"image/{fmt.lower()}",
                use_container_width=True
            )
            
        with col_ex2:
            st.write("### Bulk Processing")
            if st.button("📦 Process & Download ALL Images (ZIP)", use_container_width=True):
                zip_buffer = io.BytesIO()
                prog_bar = st.progress(0)
                status_text = st.empty()
                
                with zipfile.ZipFile(zip_buffer, "w") as zf:
                    for idx, (fname, fobj) in enumerate(file_map.items()):
                        status_text.text(f"Processing {fname}...")
                        
                        fobj.seek(0)
                        file_bytes_loop = np.asarray(bytearray(fobj.read()), dtype=np.uint8)
                        img_loop_raw = cv2.imdecode(file_bytes_loop, 1)
                        # Auto Grayscale
                        g_temp = cv2.cvtColor(img_loop_raw, cv2.COLOR_BGR2GRAY)
                        img_loop = cv2.cvtColor(g_temp, cv2.COLOR_GRAY2BGR)
                        
                        cfg_loop = get_config(fname)
                        processed_loop, _, _ = processors.master_pipeline(img_loop, cfg_loop)
                        
                        valid, buf_loop = cv2.imencode(f".{fmt.lower()}", processed_loop)
                        zf.writestr(f"restored_{fname.rsplit('.', 1)[0]}.{fmt.lower()}", buf_loop.tobytes())
                        
                        prog_bar.progress((idx + 1) / len(file_map))
                
                status_text.success("All images processed!")
                st.download_button(
                    label="⬇️ Download ZIP Archive",
                    data=zip_buffer.getvalue(),
                    file_name="restored_photos.zip",
                    mime="application/zip",
                    use_container_width=True
                )

    except Exception as e:
        st.error(f"Error processing image: {e}")

else:
    st.info("👈 Upload images in the sidebar to begin restoration.")