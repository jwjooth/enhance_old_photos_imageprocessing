"""
FILTERING MODULE
Handles noise reduction dan artifact removal

Author: Photo Enhancement Team
Version: 1.0.0
"""

import cv2
import numpy as np
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class NoiseFiltering:
    """
    Noise filtering untuk old photos:
    - Bilateral filtering (edge-preserving)
    - Non-local means denoising
    - Median filtering
    - Gaussian blur
    """
    
    def __init__(self):
        """Initialize filtering engine"""
        self.logger = logging.getLogger(__name__)
    
    def apply_filter(self, image: np.ndarray, filter_type: str = 'bilateral', strength: float = 1.0) -> Dict:
        """
        Apply selected filter ke image.
        
        Args:
            image (np.ndarray): Input image
            filter_type (str): 'bilateral', 'nlm', 'median', atau 'gaussian'
            strength (float): Filter strength (0.5 - 2.0)
            
        Returns:
            dict: {'image': filtered_image, 'filter_type': str}
        """
        
        self.logger.info(f"🧹 Applying {filter_type} filter (strength: {strength})")
        
        if filter_type == 'bilateral':
            result = self._bilateral_filter(image, strength)
        
        elif filter_type == 'nlm':
            result = self._nlm_filter(image, strength)
        
        elif filter_type == 'median':
            result = self._median_filter(image, strength)
        
        elif filter_type == 'gaussian':
            result = self._gaussian_filter(image, strength)
        
        else:
            self.logger.warning(f"Unknown filter type: {filter_type}, using bilateral")
            result = self._bilateral_filter(image, strength)
        
        return {
            'image': result,
            'filter_type': filter_type
        }
    
    def _bilateral_filter(self, image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        Bilateral filter - edge-preserving noise reduction.
        
        RECOMMENDED untuk old photos!
        
        Args:
            image (np.ndarray): Input image
            strength (float): Filter strength multiplier
            
        Returns:
            np.ndarray: Filtered image
        """
        
        # Parameter adjustments based on strength
        d = int(9 * strength)  # Diameter
        sigma_color = 75 * strength
        sigma_space = 75 * strength
        
        # Ensure odd diameter
        if d % 2 == 0:
            d += 1
        
        # Apply bilateral filter
        filtered = cv2.bilateralFilter(
            image,
            d,
            sigma_color,
            sigma_space
        )
        
        self.logger.debug(f"Bilateral filter applied (d={d}, sigma_color={sigma_color}, sigma_space={sigma_space})")
        return filtered
    
    def _nlm_filter(self, image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        Non-Local Means denoising - very effective untuk old photos.
        
        Args:
            image (np.ndarray): Input image
            strength (float): Denoising strength
            
        Returns:
            np.ndarray: Filtered image
        """
        
        # Parameters based on strength
        h = 10 * strength  # Filter strength
        template_window = 7
        search_window = 21
        
        # Check if image is color or grayscale
        if len(image.shape) == 3:
            filtered = cv2.fastNlMeansDenoisingColored(
              src=image,
              h=int(h),
              hColor=int(h),
              templateWindowSize=template_window,
              searchWindowSize=search_window,
            )
        else:
            filtered = cv2.fastNlMeansDenoising(
                image,
                h=int(h),
                templateWindowSize=template_window,
                searchWindowSize=search_window
            )
        
        self.logger.debug(f"NLM filter applied (h={h})")
        return filtered
    
    def _median_filter(self, image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        Median filtering - good untuk salt-and-pepper noise.
        
        Args:
            image (np.ndarray): Input image
            strength (float): Kernel size multiplier
            
        Returns:
            np.ndarray: Filtered image
        """
        
        # Kernel size (must be odd)
        kernel_size = int(5 * strength)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = min(kernel_size, 21)  # Cap maksimum
        
        filtered = cv2.medianBlur(image, kernel_size)
        
        self.logger.debug(f"Median filter applied (kernel_size={kernel_size})")
        return filtered
    
    def _gaussian_filter(self, image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        Gaussian blur - simple smoothing.
        
        Args:
            image (np.ndarray): Input image
            strength (float): Blur strength
            
        Returns:
            np.ndarray: Filtered image
        """
        
        # Kernel size
        kernel_size = int(5 * strength)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = min(kernel_size, 31)
        
        sigma = strength
        filtered = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        
        self.logger.debug(f"Gaussian filter applied (kernel_size={kernel_size}, sigma={sigma})")
        return filtered
    
    def combined_filter(self, image: np.ndarray, primary: str = 'bilateral', secondary: str = 'nlm', strength: float = 1.0) -> np.ndarray:
        """
        Apply combined filters untuk better results.
        
        Args:
            image (np.ndarray): Input image
            primary (str): Primary filter type
            secondary (str): Secondary filter type
            strength (float): Combined strength
            
        Returns:
            np.ndarray: Filtered image
        """
        
        self.logger.info(f"🧹 Applying combined filters: {primary} + {secondary}")
        
        # Apply primary filter
        result1 = self.apply_filter(image, primary, strength * 0.7)['image']
        
        # Apply secondary filter
        result2 = self.apply_filter(result1, secondary, strength * 0.5)['image']
        
        return result2
    
    def scratch_removal(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Specific method untuk menghilangkan scratches/goresan.
        
        Args:
            image (np.ndarray): Input image
            kernel_size (int): Morphological kernel size
            
        Returns:
            np.ndarray: Image dengan scratches berkurang
        """
        
        self.logger.info("🔧 Applying scratch removal")
        
        # Create morphological kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Apply morphological closing untuk close small holes
        result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        
        # Apply morphological opening untuk remove small noise
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
        
        return result
    
    def adaptive_filter(self, image: np.ndarray, block_size: int = 31) -> np.ndarray:
        """
        Adaptive bilateral filtering - locally adaptive strength.
        
        Args:
            image (np.ndarray): Input image
            block_size (int): Block size untuk adaptation
            
        Returns:
            np.ndarray: Filtered image
        """
        
        self.logger.info("🧹 Applying adaptive filtering")
        
        # Convert to LAB for better processing
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply bilateral filter to each channel
            l = cv2.bilateralFilter(l, 9, 50, 50)
            a = cv2.bilateralFilter(a, 9, 50, 50)
            b = cv2.bilateralFilter(b, 9, 50, 50)
            
            result = cv2.merge([l, a, b])
            result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        else:
            result = cv2.bilateralFilter(image, 9, 50, 50)
        
        return result