import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import base64
import zipfile
from streamlit_image_comparison import image_comparison
from streamlit_drawable_canvas import st_canvas

# --- CRITICAL MONKEY PATCH (ROBUST FIX) ---
# Patch ini mendefinisikan ulang fungsi image_to_url secara manual
# agar kompatibel dengan SEMUA versi Streamlit tanpa error import.
import streamlit.elements.image as st_image

# Fungsi helper mandiri untuk konversi gambar ke Base64 (Dipakai di Canvas & Patch)
def convert_image_to_base64_url(image, output_format="JPEG"):
    """
    Mengubah gambar (Numpy/PIL) menjadi URL Base64 yang valid untuk HTML/Canvas.
    """
    # Konversi Numpy ke PIL jika perlu
    if isinstance(image, np.ndarray):
        # Pastikan channel benar (BGR ke RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
    elif not isinstance(image, Image.Image):
        return ""

    # Simpan ke buffer memori
    buffered = io.BytesIO()
    try:
        image.save(buffered, format=output_format)
    except:
        image.save(buffered, format="PNG") # Fallback
        
    # Encode ke Base64 string
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/{output_format.lower()};base64,{img_str}"

if not hasattr(st_image, 'image_to_url'):
    # Terapkan fungsi kita sebagai patch jika Streamlit tidak memilikinya
    def patched_image_to_url(image, width=None, clamp=False, channels="RGB", output_format="JPEG", image_id=None):
        return convert_image_to_base64_url(image, output_format)
    
    st_image.image_to_url = patched_image_to_url
# -------------------------------------------------

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
    
    /* Status Badge Style */
    .status-badge {
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# --- STRICT PRESET DEFINITIONS ---
PRESET_LIBS = {
    "Full Body (Far / Wide Shot)": {
        'description': "Good for general cleaning without destroying small facial details.",
        'config': {
            'enable_scratch': False, 'scratch_thresh': 20,
            'bg_denoise_algo': 'Bilateral Filter', 'bg_denoise_val': 18,
            'roi_feather': 50,
            'face_denoise_algo': 'NLM (Premium)', 'face_denoise_val': 4,
            'face_enhance_algo': 'Gamma Correction', 'face_enhance_val': 38,
            'face_sharpen_algo': 'High Pass Overlay', 'face_sharpen_val': 45,
            'face_confidence': 5,
            'rotation': 0, 'zoom': 1.0, 'brightness': 0, 'contrast': 0, 'color_tone': 'B&W (Default)',
            'manual_repair_mask': None
        }
    },
    "Full Face (Close Up / Portrait)": {
        'description': "High detail retention, stronger smoothing on skin but precise edges.",
        'config': {
            'enable_scratch': False, 'scratch_thresh': 20,
            'bg_denoise_algo': 'Bilateral Filter', 'bg_denoise_val': 18,
            'roi_feather': 95,
            'face_denoise_algo': 'NLM (Premium)', 'face_denoise_val': 14,
            'face_enhance_algo': 'CLAHE', 'face_enhance_val': 50,
            'face_sharpen_algo': 'High Pass Overlay', 'face_sharpen_val': 13,
            'face_confidence': 5,
            'rotation': 0, 'zoom': 1.0, 'brightness': 0, 'contrast': 0, 'color_tone': 'B&W (Default)',
            'manual_repair_mask': None
        }
    }
}

# --- STATE MANAGEMENT INIT ---
if 'image_configs' not in st.session_state:
    st.session_state['image_configs'] = {}

# --- HELPER: GET/SET CONFIG ---
def get_default_config():
    """Returns a neutral starting configuration."""
    return {
        # Geometry & Exposure
        'rotation': 0, 'zoom': 1.0, 'brightness': 0, 'contrast': 0,
        
        # Manual Repair
        'manual_repair_mask': None, # Stores the binary mask from canvas
        
        # Restoration
        'enable_scratch': False, 'scratch_thresh': 20,
        'bg_denoise_algo': 'None', 'bg_denoise_val': 0,
        'roi_feather': 50,
        'face_denoise_algo': 'None', 'face_denoise_val': 0,
        'face_enhance_algo': 'None', 'face_enhance_val': 0,
        'face_sharpen_algo': 'None', 'face_sharpen_val': 0,
        'face_confidence': 5,
        
        # Aesthetics
        'color_tone': "B&W (Default)"
    }

def get_config(filename):
    if filename not in st.session_state['image_configs']:
        st.session_state['image_configs'][filename] = get_default_config()
    return st.session_state['image_configs'][filename]

def update_config(filename, key, value):
    st.session_state['image_configs'][filename][key] = value

def apply_strict_preset(filename, preset_name):
    if preset_name in PRESET_LIBS:
        target_conf = PRESET_LIBS[preset_name]['config']
        for key, val in target_conf.items():
            st.session_state['image_configs'][filename][key] = val

def reset_current_image(filename):
    st.session_state['image_configs'][filename] = get_default_config()

# --- MAIN UI ---
st.title("🎞️ RetroFix Pro v2.1")
st.markdown("**Rotate** • **Enhance** • **Restore** • **Tone** | Complete Photo Restoration Suite")

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
    
    # Load State for this specific image
    cfg = get_config(selected_filename)
    
    # --- PRE-PIPELINE PREVIEW (For Sidebar Stats & Canvas Base) ---
    gray_temp = cv2.cvtColor(original_loaded, cv2.COLOR_BGR2GRAY)
    base_image = cv2.cvtColor(gray_temp, cv2.COLOR_GRAY2BGR)
    
    # Apply Geometry & Exposure settings from Config
    preview_img = processors.rotate_image(base_image, cfg.get('rotation', 0))
    preview_img = processors.apply_zoom(preview_img, cfg.get('zoom', 1.0))
    preview_img = processors.adjust_brightness_contrast(preview_img, cfg.get('brightness', 0), cfg.get('contrast', 0))
    
    # Detect Faces (Just for metric)
    current_confidence = cfg.get('face_confidence', 5)
    detected_faces_preview = processors.detect_all_faces(preview_img, min_neighbors=current_confidence)
    num_faces_found = len(detected_faces_preview)
    has_faces = num_faces_found > 0

    # --- SIDEBAR CONTROLS ---
    with st.sidebar:
        # Status Metric
        col_stat1, col_stat2 = st.columns(2)
        col_stat1.metric("Color Mode", cfg.get('color_tone', 'B&W'))
        col_stat2.metric("Faces Detected", f"{num_faces_found}", delta="Ready" if has_faces else "None")
        
        st.markdown("---")
        
        # --- PRESET SYSTEM ---
        st.subheader("⚡ Quick Presets")
        preset_choice = st.selectbox("Select Mode:", list(PRESET_LIBS.keys()), index=0, key="preset_selector")
        
        col_btn1, col_btn2 = st.columns(2)
        if col_btn1.button(f"Apply Preset", use_container_width=True):
            apply_strict_preset(selected_filename, preset_choice)
            st.toast(f"Preset '{preset_choice}' applied!", icon="✅")
            st.rerun()

        if col_btn2.button("Reset All", type="primary", use_container_width=True):
            reset_current_image(selected_filename)
            st.toast("Settings reset to default.", icon="🔄") 
            st.rerun()

        st.markdown("---")
        
        # EXPANDER 0: GEOMETRY & EXPOSURE
        with st.expander("⚙️ Geometry & Exposure", expanded=False):
            st.caption("Fix orientation and lighting **before** restoration.")
            st.info("💡 **AI Tip:** Ensure faces are **upright** (vertical) for detection.")
            
            rot = st.number_input("Rotation (°)", -180, 180, cfg['rotation'], 1, help="Rotate image to fix alignment.")
            if rot != cfg['rotation']:
                update_config(selected_filename, 'rotation', rot)
                st.rerun()
                
            zoom_val = st.slider("Zoom (x)", 1.0, 2.0, cfg.get('zoom', 1.0), 0.05)
            if zoom_val != cfg.get('zoom', 1.0):
                update_config(selected_filename, 'zoom', zoom_val)
                st.rerun()

            c_bri, c_con = st.columns(2)
            bri = c_bri.slider("Brightness", -50, 50, cfg['brightness'])
            con = c_con.slider("Contrast", -50, 50, cfg['contrast'])
            update_config(selected_filename, 'brightness', bri)
            update_config(selected_filename, 'contrast', con)

        # EXPANDER: MANUAL HEALING BRUSH (NEW)
        with st.expander("🖌️ Manual Healing Brush", expanded=True):
            st.caption("Manually paint over defects.")
            show_canvas = st.toggle("Open Healing Tool", value=False, help="Switches main view to Drawing Mode")
            
            if show_canvas:
                stroke_width = st.slider("Brush Size", 1, 25, 10)
                if st.button("🗑️ Clear Manual Repairs", use_container_width=True):
                    update_config(selected_filename, 'manual_repair_mask', None)
                    st.rerun()

        # EXPANDER 1: REPAIR & SCRATCHES
        with st.expander("🛠️ Auto-Repair (BG)", expanded=False):
            enable_scratch = st.toggle("Enable Smart Scratch Removal", value=cfg['enable_scratch'])
            update_config(selected_filename, 'enable_scratch', enable_scratch)
            
            if enable_scratch:
                scratch_thresh = st.slider("Scratch Threshold", 0, 255, cfg['scratch_thresh'])
                update_config(selected_filename, 'scratch_thresh', scratch_thresh)
                show_scratch_mask = st.toggle("Show Red Scratch Mask", value=False)
            else:
                show_scratch_mask = False
            
            st.markdown("---")
            bg_denoise_algo = st.selectbox("BG Denoise Algo", ["None", "Gaussian Blur", "Median Blur", "Bilateral Filter"], index=["None", "Gaussian Blur", "Median Blur", "Bilateral Filter"].index(cfg['bg_denoise_algo']))
            update_config(selected_filename, 'bg_denoise_algo', bg_denoise_algo)
            bg_denoise_val = st.slider("BG Denoise Strength", 0, 30, cfg['bg_denoise_val'])
            update_config(selected_filename, 'bg_denoise_val', bg_denoise_val)

        # EXPANDER 2: FACE / ROI DETAILS
        with st.expander("👤 Face / ROI Details", expanded=False):
            if not has_faces:
                st.warning("⚠️ No faces detected. Try Rotate or Brightness.")
            
            st.markdown("#### 🕵️ Detection")
            face_confidence = st.slider("Confidence", 1, 10, int(current_confidence))
            if face_confidence != current_confidence:
                update_config(selected_filename, 'face_confidence', face_confidence)
                st.rerun()
                
            st.markdown("#### ✨ Enhancement")
            roi_feather = st.slider("Feathering", 1, 150, cfg['roi_feather'], step=2)
            update_config(selected_filename, 'roi_feather', roi_feather)
            
            f_denoise = st.selectbox("Face Denoise", ["None", "NLM (Premium)", "Bilateral Filter"], index=["None", "NLM (Premium)", "Bilateral Filter"].index(cfg['face_denoise_algo']))
            update_config(selected_filename, 'face_denoise_algo', f_denoise)
            if f_denoise != "None":
                fd_val = st.slider("Level", 0, 30, cfg['face_denoise_val'])
                update_config(selected_filename, 'face_denoise_val', fd_val)

            f_enhance = st.selectbox("Face Enhance", ["None", "CLAHE", "Histogram Eq", "Gamma Correction"], index=["None", "CLAHE", "Histogram Eq", "Gamma Correction"].index(cfg['face_enhance_algo']))
            update_config(selected_filename, 'face_enhance_algo', f_enhance)
            if f_enhance != "None":
                fe_val = st.slider("Enhance Lvl", 0, 100, cfg['face_enhance_val'])
                update_config(selected_filename, 'face_enhance_val', fe_val)
                
            f_sharpen = st.selectbox("Face Sharpening", ["None", "Unsharp Masking", "Laplacian", "High Pass Overlay"], index=["None", "Unsharp Masking", "Laplacian", "High Pass Overlay"].index(cfg['face_sharpen_algo']))
            update_config(selected_filename, 'face_sharpen_algo', f_sharpen)
            if f_sharpen != "None":
                fs_val = st.slider("Amount", 0, 100, cfg['face_sharpen_val'])
                update_config(selected_filename, 'face_sharpen_val', fs_val)

        # EXPANDER 3: AESTHETICS (NEW)
        with st.expander("🎨 Final Touch (Toning)", expanded=False):
            tone = st.selectbox("Color Tone", ["B&W (Default)", "Sepia (Vintage)", "Selenium (Cool)"], index=["B&W (Default)", "Sepia (Vintage)", "Selenium (Cool)"].index(cfg['color_tone']))
            update_config(selected_filename, 'color_tone', tone)

    # --- MAIN VIEW SWITCHER ---
    if show_canvas:
        # --- CANVAS MODE (ROBUST & RESIZED) ---
        st.subheader("🖌️ Manual Healing Mode")
        st.info("Paint over tears, spots, or creases. The app will auto-fill them.")
        
        # 1. Resize Preview Image for Canvas (Agar muat di layar dan tidak blank)
        # Gunakan 'preview_img' yang sudah di-rotate & brightness
        max_canvas_width = 800
        orig_h, orig_w = preview_img.shape[:2]
        
        if orig_w > max_canvas_width:
            scale_factor = max_canvas_width / orig_w
            display_w = max_canvas_width
            display_h = int(orig_h * scale_factor)
            # Resize untuk display
            canvas_bg_display = cv2.resize(preview_img, (display_w, display_h), interpolation=cv2.INTER_AREA)
        else:
            scale_factor = 1.0
            canvas_bg_display = preview_img
            
        # 2. Convert to PIL Image (AGAR ST_CANVAS TIDAK ERROR)
        # st_canvas butuh objek PIL Image jika kita kirim argumen height/width
        pil_image = Image.fromarray(cv2.cvtColor(canvas_bg_display, cv2.COLOR_BGR2RGB))
        
        # 3. Create Canvas
        # PENTING: Gunakan 'pil_image' (Objek), bukan base64 string.
        # Patch di atas (st_image.image_to_url) akan menangani konversinya.
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",  # Transparan Merah
            stroke_width=stroke_width,
            stroke_color="rgba(255, 0, 0, 1)", # Solid Merah
            background_image=pil_image, # << Kembali ke PIL Object
            update_streamlit=True,
            height=canvas_bg_display.shape[0], 
            width=canvas_bg_display.shape[1],
            drawing_mode="freedraw",
            key="canvas",
        )
        
        # 4. Handle Output
        if canvas_result.image_data is not None:
            # Mask yang dihasilkan canvas berukuran KECIL (sesuai display_w/h)
            # Kita simpan apa adanya. 
            # Processors.py sudah punya logika resize otomatis (apply_manual_repair)
            # yang akan membesarkan mask ini kembali ke ukuran asli gambar.
            update_config(selected_filename, 'manual_repair_mask', canvas_result.image_data)
            
            # Show Realtime Preview 
            if st.session_state['image_configs'][selected_filename]['manual_repair_mask'] is not None:
                st.caption("Real-time Patch Preview:")
                # Untuk preview cepat, kita pakai gambar display saja
                patched_display = processors.apply_manual_repair(
                    canvas_bg_display.copy(), 
                    st.session_state['image_configs'][selected_filename]['manual_repair_mask']
                )
                st.image(cv2.cvtColor(patched_display, cv2.COLOR_BGR2RGB), use_container_width=True)

    else:
        # --- RESTORATION MODE (DEFAULT) ---
        try:
            with st.spinner("Processing Restoration..."):
                # Master pipeline now handles everything including Manual Mask
                final_img_bgr, roi_mask, scratch_mask_vis = processors.master_pipeline(original_loaded, cfg)
            
            # Display Setup
            final_img_rgb = cv2.cvtColor(final_img_bgr, cv2.COLOR_BGR2RGB)
            
            # For "Before" image, we show the ROTATED/ZOOMED/EXPOSED base
            # This ensures 'Before' and 'After' are perfectly aligned
            original_img_rgb = cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB)
            
            # Visual Debug Overlay for Auto Scratch
            if show_scratch_mask and cfg['enable_scratch']:
                red_mask = np.zeros_like(final_img_rgb)
                red_mask[:,:,0] = 255 
                mask_indices = scratch_mask_vis > 0
                final_img_rgb[mask_indices] = cv2.addWeighted(final_img_rgb[mask_indices], 0.7, red_mask[mask_indices], 0.3, 0)

            # 1. Debug Mask Visualization
            with st.expander("👁️ View Masks (ROI & Scratch)", expanded=False):
                dc1, dc2 = st.columns(2)
                dc1.image(roi_mask, caption=f"Faces ({num_faces_found})", use_container_width=True, clamp=True)
                if cfg['enable_scratch']:
                    dc2.image(scratch_mask_vis, caption="Auto-Scratch Mask", use_container_width=True)
                else:
                    dc2.info("Auto-Scratch Disabled")

            # 2. Juxtapose Viewer
            st.subheader(f"Result: {cfg.get('color_tone', 'B&W')}")
            image_comparison(
                img1=original_img_rgb,
                img2=final_img_rgb,
                label1="Base (Aligned)",
                label2="Restored",
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
                    label="⬇️ Download Image",
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
                            cfg_loop = get_config(fname)
                            
                            processed_loop, _, _ = processors.master_pipeline(img_loop_raw, cfg_loop)
                            
                            valid, buf_loop = cv2.imencode(f".{fmt.lower()}", processed_loop)
                            zf.writestr(f"restored_{fname.rsplit('.', 1)[0]}.{fmt.lower()}", buf_loop.tobytes())
                            
                            prog_bar.progress((idx + 1) / len(file_map))
                    
                    status_text.success("All images processed!")
                    st.download_button(
                        label="⬇️ Download ZIP",
                        data=zip_buffer.getvalue(),
                        file_name="restored_photos.zip",
                        mime="application/zip",
                        use_container_width=True
                    )

        except Exception as e:
            st.error(f"Error processing image: {e}")

else:
    st.info("👈 Upload images to begin.")