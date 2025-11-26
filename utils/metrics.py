import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ImageMetrics:
    """Calculate quality metrics untuk enhanced images"""
    
    @staticmethod
    def calculate_sharpness(image):
        """
        Calculate sharpness menggunakan Laplacian variance.
        Higher = sharper
        
        Args:
            image (np.ndarray): Image
            
        Returns:
            float: Sharpness score
        """
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        return sharpness
    
    @staticmethod
    def calculate_contrast(image):
        """
        Calculate contrast menggunakan standard deviation.
        
        Args:
            image (np.ndarray): Image
            
        Returns:
            float: Contrast score
        """
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        contrast = gray.std()
        return contrast
    
    @staticmethod
    def calculate_brightness(image):
        """
        Calculate average brightness.
        
        Args:
            image (np.ndarray): Image
            
        Returns:
            float: Brightness value (0-255)
        """
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        brightness = gray.mean()
        return brightness
    
    @staticmethod
    def compare_images(original, enhanced):
        """
        Compare original vs enhanced image.
        
        Args:
            original (np.ndarray): Original image
            enhanced (np.ndarray): Enhanced image
            
        Returns:
            dict: Comparison metrics
        """
        
        metrics = {
            'original_sharpness': ImageMetrics.calculate_sharpness(original),
            'enhanced_sharpness': ImageMetrics.calculate_sharpness(enhanced),
            'original_contrast': ImageMetrics.calculate_contrast(original),
            'enhanced_contrast': ImageMetrics.calculate_contrast(enhanced),
            'original_brightness': ImageMetrics.calculate_brightness(original),
            'enhanced_brightness': ImageMetrics.calculate_brightness(enhanced),
        }
        
        # Calculate improvements
        metrics['sharpness_improvement'] = (
            (metrics['enhanced_sharpness'] - metrics['original_sharpness']) /
            metrics['original_sharpness'] * 100
        )
        
        metrics['contrast_improvement'] = (
            (metrics['enhanced_contrast'] - metrics['original_contrast']) /
            metrics['original_contrast'] * 100
        )
        
        return metrics