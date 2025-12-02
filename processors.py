import cv2
import numpy as np
import os

def rotate_image(image, angle):
    if angle == 0:
        return image
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    rotated = cv2.warpAffine(image, M, (nW, nH), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    return rotated

def apply_zoom(img, zoom_factor):
    if zoom_factor <= 1.0:
        return img
    h, w = img.shape[:2]
    new_w = int(w / zoom_factor)
    new_h = int(h / zoom_factor)
    start_x = (w - new_w) // 2
    start_y = (h - new_h) // 2
    cropped = img[start_y:start_y+new_h, start_x:start_x+new_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

def adjust_brightness_contrast(img, brightness=0, contrast=0):
    if brightness == 0 and contrast == 0:
        return img
    alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
    beta = brightness
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted

def apply_color_tone(img, tone_name):
    if tone_name == "B&W (Default)":
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_3c = cv2.merge([gray, gray, gray])
    if tone_name == "Sepia (Vintage)":
        b, g, r = cv2.split(gray_3c)
        b = np.clip(b * 0.8, 0, 255).astype(np.uint8)
        g = np.clip(g * 0.95, 0, 255).astype(np.uint8)
        r = np.clip(r * 1.1, 0, 255).astype(np.uint8)
        return cv2.merge([b, g, r])
    elif tone_name == "Selenium (Cool)":
        b, g, r = cv2.split(gray_3c)
        b = np.clip(b * 1.1, 0, 255).astype(np.uint8)
        g = np.clip(g * 1.0, 0, 255).astype(np.uint8)
        r = np.clip(r * 0.9, 0, 255).astype(np.uint8)
        return cv2.merge([b, g, r])
    return img

def create_multi_face_mask(image_shape, faces, feather_amount):
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    for (cx, cy, rx, ry) in faces:
        center = (int(cx * w), int(cy * h))
        axes = (int(rx * w), int(ry * h))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    ksize = int(feather_amount)
    if ksize % 2 == 0: ksize += 1
    if ksize > 0: mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
    return mask

def detect_all_faces(img, min_neighbors=5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces_rects = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=int(min_neighbors), minSize=(30, 30)
    )
    if len(faces_rects) == 0: return []
    normalized_faces = []
    h_img, w_img = img.shape[:2]
    for (x, y, w, h) in faces_rects:
        center_x = (x + w/2) / w_img
        center_y = (y + h/2) / h_img
        radius_x = (w * 0.6) / w_img 
        radius_y = (h * 0.6) / h_img
        normalized_faces.append((center_x, center_y, radius_x, radius_y))
    return normalized_faces

def apply_denoise(img, method, strength):
    if method == "None": return img
    strength = int(strength)
    if strength == 0: return img
    if method == "Gaussian Blur":
        k = strength | 1
        return cv2.GaussianBlur(img, (k, k), 0)
    elif method == "Median Blur":
        k = strength | 1
        if k % 2 == 0: k += 1
        return cv2.medianBlur(img, k)
    elif method == "Bilateral Filter":
        return cv2.bilateralFilter(img, 9, strength * 2, strength / 2)
    elif method == "NLM (Premium)":
        h = strength / 2.0
        return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)
    return img

def apply_enhancement(img, method, level):
    if method == "None": return img
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    if method == "Histogram Eq": l = cv2.equalizeHist(l)
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
    if method == "None" or amount == 0: return img
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
    
    face_binary = (mask_protector > 0.1).astype(np.uint8) * 255
    safe_zone = cv2.bitwise_not(face_binary)
    final_inpaint_mask = cv2.bitwise_and(scratch_mask, safe_zone)
    restored = cv2.inpaint(img, final_inpaint_mask, 3, cv2.INPAINT_TELEA)
    return restored, final_inpaint_mask

def apply_manual_repair(img, mask_rgba):
    """
    Apply manual inpainting based on user drawing from st_canvas.
    mask_rgba: Image data from canvas (Usually RGBA uint8).
    """
    if mask_rgba is None:
        return img
    
    # Convert RGBA Mask from Canvas to Single Channel Binary
    # Canvas returns RGBA where drawn pixels have alpha > 0
    if len(mask_rgba.shape) == 3 and mask_rgba.shape[2] == 4:
        # Ambil channel alpha sebagai mask
        mask = mask_rgba[:, :, 3]
    else:
        mask = mask_rgba
    
    # Ensure binary mask
    _, binary_mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    
    # Safety: Resize mask to match image if dimensions differ (UI scaling issue prevention)
    if binary_mask.shape[:2] != img.shape[:2]:
        binary_mask = cv2.resize(binary_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Inpaint
    # Radius 3 is standard for Telea
    restored = cv2.inpaint(img, binary_mask, 3, cv2.INPAINT_TELEA)
    return restored

def master_pipeline(original_img, config):
    """
    Pipeline Lengkap:
    Raw -> Grayscale -> Rotate -> ZOOM -> Bright/Contrast -> [MANUAL REPAIR] -> Detect Face -> Restorasi -> Toning
    """
    # 1. Force Grayscale Foundation
    gray_base = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(gray_base, cv2.COLOR_GRAY2BGR)
    
    # 2. Geometry: Rotation
    rot_angle = config.get('rotation', 0)
    img = rotate_image(img, rot_angle)
    
    # 3. Geometry: Zoom
    zoom_factor = config.get('zoom', 1.0)
    img = apply_zoom(img, zoom_factor)
    
    # 4. Exposure: Brightness & Contrast
    bri = config.get('brightness', 0)
    con = config.get('contrast', 0)
    img = adjust_brightness_contrast(img, brightness=bri, contrast=con)
    
    # --- MANUAL REPAIR STEP (NEW) ---
    # Dilakukan sebelum deteksi wajah agar wajah yang diperbaiki terdeteksi
    manual_mask = config.get('manual_repair_mask', None)
    if manual_mask is not None:
        img = apply_manual_repair(img, manual_mask)
    
    h, w = img.shape[:2]
    
    # 5. Detect ALL Faces automatically
    confidence = config.get('face_confidence', 5)
    faces = detect_all_faces(img, min_neighbors=confidence)
    
    # 6. Mask Creation
    roi_mask = create_multi_face_mask(img.shape, faces, config['roi_feather'])
    roi_mask_3c = cv2.merge([roi_mask, roi_mask, roi_mask])
    bg_mask_3c = 1.0 - roi_mask_3c
    
    # --- PIPELINE BACKGROUND ---
    bg_processed = img.copy()
    scratch_mask_vis = None
    if config['enable_scratch']:
        bg_processed, scratch_mask_vis = apply_scratch_removal(bg_processed, config['scratch_thresh'], roi_mask)
    else:
        scratch_mask_vis = np.zeros((h, w), dtype=np.uint8)

    bg_processed = apply_denoise(bg_processed, config['bg_denoise_algo'], config['bg_denoise_val'])
    
    # --- PIPELINE FACE / ROI ---
    face_processed = img.copy() 
    face_processed = apply_denoise(face_processed, config['face_denoise_algo'], config['face_denoise_val'])
    face_processed = apply_enhancement(face_processed, config['face_enhance_algo'], config['face_enhance_val'])
    face_processed = apply_sharpening(face_processed, config['face_sharpen_algo'], config['face_sharpen_val'])
    
    # --- BLENDING ---
    final_img = (face_processed.astype(float) * roi_mask_3c) + (bg_processed.astype(float) * bg_mask_3c)
    final_img = np.clip(final_img, 0, 255).astype(np.uint8)
    
    # 7. Final Touch: Color Toning
    final_tone = config.get('color_tone', "B&W (Default)")
    final_img = apply_color_tone(final_img, final_tone)
    
    return final_img, roi_mask, scratch_mask_vis