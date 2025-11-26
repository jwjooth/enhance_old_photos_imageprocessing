"""
HISTOGRAM EQUALIZATION MODULE
Handles contrast enhancement dan brightness restoration

Author: Photo Enhancement Team
Version: 1.0.0
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class HistogramEqualization:
    """
    Histogram equalization untuk old photos:
    - CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - Standard histogram equalization
    - Multi-scale equalization
    """
    
    def __init__(self):
        """Initialize histogram equalization engine"""
        self.logger = logging.getLogger(__name__)
    
    def enhance(self, image, method='clahe', clip_limit=2.0):
        """
        Main enhancement function.
        
        Args:
            image (np.ndarray): Input image
            method (str): 'clahe', 'standard', 'multiscale'
            clip_limit (float): Contrast limit (1.0 - 4.0)
            
        Returns:
            dict: {'image': enhanced_image, 'method': str}
        """
        
        self.logger.info(f"💡 Enhancing with {method} (clip_limit: {clip_limit})")
        
        if method == 'clahe':
            result = self._clahe_enhance(image, clip_limit)
        
        elif method == 'standard':
            result = self._standard_equalization(image)
        
        elif method == 'multiscale':
            result = self._multiscale_enhance(image, clip_limit)
        
        else:
            self.logger.warning(f"Unknown method: {method}, using clahe")
            result = self._clahe_enhance(image, clip_limit)
        
        return {
            'image': result,
            'method': method
        }
    
    def _clahe_enhance(self, image, clip_limit=2.0):
        """
        CLAHE - Contrast Limited Adaptive Histogram Equalization.
        RECOMMENDED untuk old photos!
        
        Args:
            image (np.ndarray): Input image
            clip_limit (float): Contrast limit (1.0 - 4.0)
            
        Returns:
            np.ndarray: Enhanced image
        """
        
        # Create CLAHE object
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(8, 8)
        )
        
        # Convert to LAB color space (better perceptual results)
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE hanya ke L (lightness) channel
            l = clahe.apply(l)
            
            # Merge channels kembali
            enhanced_lab = cv2.merge([l, a, b])
            result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        else:
            result = clahe.apply(image)
        
        self.logger.debug(f"CLAHE applied (clipLimit={clip_limit})")
        return result
    
    def _standard_equalization(self, image):
        """
        Standard histogram equalization.
        Simple tetapi bisa menghasilkan over-contrast.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            np.ndarray: Enhanced image
        """
        
        if len(image.shape) == 3:
            # Color image
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            # Equalize V channel (brightness)
            v = cv2.equalizeHist(v)
            
            enhanced_hsv = cv2.merge([h, s, v])
            result = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
        else:
            # Grayscale image
            result = cv2.equalizeHist(image)
        
        self.logger.debug("Standard histogram equalization applied")
        return result
    
    def _multiscale_enhance(self, image, clip_limit=2.0):
        """
        Multi-scale enhancement untuk better results.
        Combine CLAHE dengan standard equalization.
        
        Args:
            image (np.ndarray): Input image
            clip_limit (float): Contrast limit
            
        Returns:
            np.ndarray: Enhanced image
        """
        
        # Apply CLAHE
        clahe_result = self._clahe_enhance(image, clip_limit * 0.7)
        
        # Apply standard equalization untuk boost
        lab = cv2.cvtColor(clahe_result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply limited standard equalization
        l_eq = cv2.equalizeHist(l)
        
        # Blend 70% original, 30% equalized
        l_blend = cv2.addWeighted(l, 0.7, l_eq, 0.3, 0)
        
        result_lab = cv2.merge([l_blend, a, b])
        result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
        
        self.logger.debug("Multi-scale enhancement applied")
        return result
    
    def automatic_brightness_contrast(self, image):
        """
        Automatically adjust brightness dan contrast.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            np.ndarray: Adjusted image
        """
        
        self.logger.info("🔧 Applying automatic brightness/contrast adjustment")
        
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Calculate mean brightness
        l_mean = np.mean(l)
        l_std = np.std(l)
        
        # Target values
        target_mean = 127
        target_std = 60
        
        # Adjust
        if l_std > 0:
            l_adjusted = ((l - l_mean) * (target_std / l_std) + target_mean).astype(np.uint8)
        else:
            l_adjusted = l
        
        # Ensure values dalam range
        l_adjusted = np.clip(l_adjusted, 0, 255).astype(np.uint8)
        
        result_lab = cv2.merge([l_adjusted, a, b])
        result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
        
        return result
    
    def gamma_correction(self, image, gamma=1.2):
        """
        Apply gamma correction untuk brighten atau darken image.
        
        Args:
            image (np.ndarray): Input image
            gamma (float): Gamma value
                          > 1.0 = darken
                          < 1.0 = brighten
            
        Returns:
            np.ndarray: Gamma-corrected image
        """
        
        self.logger.info(f"🔧 Applying gamma correction (gamma={gamma})")
        
        # Build lookup table
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in np.arange(0, 256)]).astype(np.uint8)
        
        # Apply lookup table
        result = cv2.LUT(image, table)
        
        return result
    
    def color_balance(self, image):
        """
        Improve color balance untuk old photos yang color-faded.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            np.ndarray: Color-balanced image
        """
        
        self.logger.info("🎨 Applying color balance")
        
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE ke a dan b channels (color channels)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        a = clahe.apply(a)
        b = clahe.apply(b)
        
        result_lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
        
        return result
    
    def local_contrast_enhancement(self, image, strength=1.5):
        """
        Enhance local contrast menggunakan Laplacian.
        
        Args:
            image (np.ndarray): Input image
            strength (float): Enhancement strength
            
        Returns:
            np.ndarray: Enhanced image
        """
        
        self.logger.info(f"💡 Applying local contrast enhancement (strength={strength})")
        
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Calculate Laplacian
        laplacian = cv2.Laplacian(l, cv2.CV_32F)
        
        # Add to original
        l_float = l.astype(np.float32)
        enhanced_l = l_float + laplacian * strength
        
        # Clip dan convert back
        enhanced_l = np.clip(enhanced_l, 0, 255).astype(np.uint8)
        
        result_lab = cv2.merge([enhanced_l, a, b])
        result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
        
        return result