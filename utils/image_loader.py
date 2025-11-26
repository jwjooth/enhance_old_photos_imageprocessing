import cv2
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ImageLoader:
    """Handle image loading dan validation"""
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    MAX_SIZE = (4096, 4096)
    
    @staticmethod
    def load_image(image_path, max_size=None):
        """
        Load image dengan validation.
        
        Args:
            image_path (str): Path ke image
            max_size (tuple): Max (width, height)
            
        Returns:
            np.ndarray: Image atau None jika gagal
        """
        
        try:
            path = Path(image_path)
            
            # Check file exists
            if not path.exists():
                logger.error(f"File not found: {image_path}")
                return None
            
            # Check format
            if path.suffix.lower() not in ImageLoader.SUPPORTED_FORMATS:
                logger.error(f"Unsupported format: {path.suffix}")
                return None
            
            # Load image
            image = cv2.imread(str(image_path))
            
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
            
            # Resize jika terlalu besar
            if max_size:
                h, w = image.shape[:2]
                if w > max_size[0] or h > max_size[1]:
                    scale = min(max_size[0]/w, max_size[1]/h)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    image = cv2.resize(image, (new_w, new_h))
                    logger.info(f"Image resized to {new_w}x{new_h}")
            
            logger.info(f"✓ Image loaded: {image.shape}")
            return image
            
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return None
    
    @staticmethod
    def save_image(image, output_path, quality=95):
        """
        Save image dengan quality control.
        
        Args:
            image (np.ndarray): Image to save
            output_path (str): Output path
            quality (int): JPEG quality (1-100)
            
        Returns:
            bool: Success or failure
        """
        
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save dengan quality
            ext = output_path.suffix.lower()
            if ext in ['.jpg', '.jpeg']:
                cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            else:
                cv2.imwrite(str(output_path), image)
            
            logger.info(f"✓ Image saved: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving image: {e}")
            return False