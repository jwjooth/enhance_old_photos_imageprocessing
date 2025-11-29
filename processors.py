import cv2
import numpy as np
import os

def create_multi_face_mask(image_shape, faces, feather_amount):
    """
    Membuat masker gabungan untuk BANYAK wajah sekaligus.
    faces: List of tuples (center_x, center_y, radius_x, radius_y) normalized.
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    
    # Loop semua wajah yang terdeteksi dan gambar ke mask
    for (cx, cy, rx, ry) in faces:
        center = (int(cx * w), int(cy * h))
        axes = (int(rx * w), int(ry * h))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    
    # Apply Gaussian Blur untuk feathering (sekali saja di akhir)
    ksize = int(feather_amount)
    if ksize % 2 == 0:
        ksize += 1
    if ksize > 0:
        mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
        
    return mask

def detect_all_faces(img):
    """
    Mendeteksi SEMUA wajah dalam gambar.
    Returns: List of (center_x, center_y, radius_x, radius_y) normalized 0.0-1.0
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Load Haar Cascade
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    # Detect faces
    faces_rects = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(30, 30)
    )
    
    if len(faces_rects) == 0:
        return []
        
    normalized_faces = []
    h_img, w_img = img.shape[:2]
    
    for (x, y, w, h) in faces_rects:
        # Konversi bbox (x, y, w, h) ke format Ellipse ROI (cx, cy, rx, ry) normalized
        center_x = (x + w/2) / w_img
        center_y = (y + h/2) / h_img
        
        # Radius dengan padding sedikit (1.2x)
        radius_x = (w * 0.6) / w_img 
        radius_y = (h * 0.6) / h_img
        
        normalized_faces.append((center_x, center_y, radius_x, radius_y))
    
    return normalized_faces

def apply_denoise(img, method, strength):
    """Menerapkan algoritma denoising."""
    if method == "None":
        return img
    
    strength = int(strength)
    if strength == 0:
        return img

    if method == "Gaussian Blur":
        k = strength | 1  # Ensure odd
        return cv2.GaussianBlur(img, (k, k), 0)
    
    elif method == "Median Blur":
        k = strength | 1
        if k % 2 == 0: k += 1
        return cv2.medianBlur(img, k)
    
    elif method == "Bilateral Filter":
        return cv2.bilateralFilter(img, 9, strength * 2, strength / 2)
    
    elif method == "NLM (Premium)":
        h = strength / 2.0
        # Menggunakan NLM versi grayscale jika input grayscale, tapi 
        # karena pipeline kita memaksakan format BGR (walau isinya gray),
        # kita tetap pakai Colored atau standar.
        return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)
        
    return img

def apply_enhancement(img, method, level):
    if method == "None":
        return img

    # Convert ke LAB tidak ideal untuk pure Grayscale, 
    # tapi karena input kita BGR (Gray content), ini tetap jalan.
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    if method == "Histogram Eq":
        l = cv2.equalizeHist(l)
        
    elif method == "CLAHE":
        clip_limit = max(1.0, level / 20.0) 
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
    elif method == "Gamma Correction":
        gamma = level / 50.0 
        if gamma == 0: gamma = 0.1
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        l = cv2.LUT(l, table)

    merged = cv2.merge((l, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def apply_sharpening(img, method, amount):
    if method == "None" or amount == 0:
        return img

    if method == "Unsharp Masking":
        gaussian = cv2.GaussianBlur(img, (0, 0), 2.0)
        alpha = 1.0 + (amount / 50.0) 
        return cv2.addWeighted(img, alpha, gaussian, - (amount / 50.0), 0)
    
    elif method == "Laplacian":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(img, -1, kernel)
        blend_val = amount / 100.0
        return cv2.addWeighted(img, 1.0 - blend_val, sharpened, blend_val, 0)
    
    elif method == "High Pass Overlay":
        blur = cv2.GaussianBlur(img, (21, 21), 0)
        high_pass = img.astype(np.int16) - blur.astype(np.int16)
        scale = amount / 50.0
        sharpened = img.astype(np.float32) + (high_pass * scale)
        return np.clip(sharpened, 0, 255).astype(np.uint8)
        
    return img

def apply_scratch_removal(img, threshold, mask_protector):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, scratch_mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)
    scratch_mask = cv2.dilate(scratch_mask, None)
    
    # mask_protector is float 0-1. White=Face.
    # Convert face mask to binary to protect it.
    face_binary = (mask_protector > 0.1).astype(np.uint8) * 255
    safe_zone = cv2.bitwise_not(face_binary) # Background area
    
    final_inpaint_mask = cv2.bitwise_and(scratch_mask, safe_zone)
    restored = cv2.inpaint(img, final_inpaint_mask, 3, cv2.INPAINT_TELEA)
    return restored, final_inpaint_mask

def master_pipeline(original_img, config):
    """
    Pipeline Utama.
    1. Konversi ke Grayscale (Pondasi Kokoh).
    2. Deteksi SEMUA wajah.
    3. Proses restorasi.
    """
    # 1. Force Grayscale Foundation
    # Convert to Gray then back to BGR to ensure 3 channels for pipeline compatibility
    gray_base = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(gray_base, cv2.COLOR_GRAY2BGR)
    
    h, w = img.shape[:2]
    
    # 2. Detect ALL Faces automatically
    # Kita tidak lagi pakai manual sliders (roi_x, roi_y)
    faces = detect_all_faces(img)
    
    # 3. Buat Masker untuk SEMUA wajah
    roi_mask = create_multi_face_mask(img.shape, faces, config['roi_feather'])
    
    roi_mask_3c = cv2.merge([roi_mask, roi_mask, roi_mask])
    bg_mask_3c = 1.0 - roi_mask_3c
    
    # --- PIPELINE BACKGROUND ---
    bg_processed = img.copy()
    
    # A. Scratch Removal
    scratch_mask_vis = None
    if config['enable_scratch']:
        bg_processed, scratch_mask_vis = apply_scratch_removal(
            bg_processed, 
            config['scratch_thresh'], 
            roi_mask
        )
    else:
        scratch_mask_vis = np.zeros((h, w), dtype=np.uint8)

    # B. Denoise BG
    bg_processed = apply_denoise(bg_processed, config['bg_denoise_algo'], config['bg_denoise_val'])
    
    # --- PIPELINE FACE / ROI ---
    face_processed = img.copy() 
    
    face_processed = apply_denoise(face_processed, config['face_denoise_algo'], config['face_denoise_val'])
    face_processed = apply_enhancement(face_processed, config['face_enhance_algo'], config['face_enhance_val'])
    face_processed = apply_sharpening(face_processed, config['face_sharpen_algo'], config['face_sharpen_val'])
    
    # --- BLENDING ---
    final_img = (face_processed.astype(float) * roi_mask_3c) + (bg_processed.astype(float) * bg_mask_3c)
    final_img = np.clip(final_img, 0, 255).astype(np.uint8)
    
    return final_img, roi_mask, scratch_mask_vis