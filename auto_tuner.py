import cv2
import numpy as np

def analyze_and_recommend(image):
    """
    Menganalisis statistik gambar dan mengembalikan rekomendasi parameter.
    Safety Rule: TIDAK menyarankan Scratch Removal.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Hitung variance dan std deviation
    mean, std_dev = cv2.meanStdDev(gray)
    variance = std_dev[0][0] ** 2
    
    # Deteksi "Flatness" / "Blurriness" menggunakan Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    recommendations = {}
    
    # 1. Rekomendasi Sharpening (Face)
    # Jika gambar buram (laplacian var rendah), butuh sharpening tinggi
    if laplacian_var < 100:
        recommendations['face_sharpen_algo'] = "Unsharp Masking"
        recommendations['face_sharpen_val'] = 60 # Kuat
    elif laplacian_var < 300:
        recommendations['face_sharpen_algo'] = "High Pass Overlay"
        recommendations['face_sharpen_val'] = 30 # Sedang
    else:
        recommendations['face_sharpen_algo'] = "None"
        recommendations['face_sharpen_val'] = 0

    # 2. Rekomendasi Denoise (Background & Face)
    # Jika variance local tinggi tapi bukan edge (heuristic kasar), mungkin noise
    # Kita pakai preset aman saja berdasarkan asumsi foto tua
    recommendations['bg_denoise_algo'] = "Median Blur" # Bagus untuk salt-pepper foto tua
    recommendations['bg_denoise_val'] = 3
    
    recommendations['face_denoise_algo'] = "NLM (Premium)" # Wajah butuh kualitas
    recommendations['face_denoise_val'] = 5

    # 3. Rekomendasi Contrast / CLAHE
    # Cek histogram spread. Jika std_dev rendah, berarti kontras rendah.
    if std_dev[0][0] < 40:
        recommendations['face_enhance_algo'] = "CLAHE"
        recommendations['face_enhance_val'] = 40 # Boost contrast
    else:
        recommendations['face_enhance_algo'] = "Gamma Correction"
        recommendations['face_enhance_val'] = 50 # Neutral (1.0 gamma)

    return recommendations

def apply_preset(preset_name):
    """Mengembalikan dictionary config lengkap berdasarkan preset."""
    # Base reset config
    config = {
        # ROI Default Center
        'roi_x': 0.5, 'roi_y': 0.5, 'roi_rx': 0.2, 'roi_ry': 0.3, 'roi_feather': 51,
        
        # Scratch
        'enable_scratch': False, 'scratch_thresh': 20,
        
        # BG
        'bg_denoise_algo': 'None', 'bg_denoise_val': 0,
        
        # Face
        'face_denoise_algo': 'None', 'face_denoise_val': 0,
        'face_enhance_algo': 'None', 'face_enhance_val': 0,
        'face_sharpen_algo': 'None', 'face_sharpen_val': 0
    }
    
    if preset_name == "Portrait Mode":
        config.update({
            'bg_denoise_algo': 'Gaussian Blur', 'bg_denoise_val': 5,
            'face_denoise_algo': 'NLM (Premium)', 'face_denoise_val': 5,
            'face_sharpen_algo': 'Unsharp Masking', 'face_sharpen_val': 30,
            'face_enhance_algo': 'CLAHE', 'face_enhance_val': 25
        })
    elif preset_name == "Old Photo Cleanup":
        config.update({
            'enable_scratch': True, 'scratch_thresh': 15,
            'bg_denoise_algo': 'Median Blur', 'bg_denoise_val': 3,
            'face_enhance_algo': 'Gamma Correction', 'face_enhance_val': 60 # Slightly brighter
        })
        
    return config