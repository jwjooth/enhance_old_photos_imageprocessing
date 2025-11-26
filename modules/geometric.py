"""
GEOMETRIC CORRECTION MODULE
Handles rotation, perspective correction, and geometric transformations

Author: Photo Enhancement Team
Version: 1.0.0
"""

import cv2
import numpy as np
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class GeometricCorrection:
    """
    Geometric correction untuk old photos:
    - Auto-rotation detection & correction
    - Perspective transformation
    - Skew correction
    """
    
    def __init__(self):
        """Initialize geometric correction engine"""
        self.logger = logging.getLogger(__name__)
    
    def correct(self, image: np.ndarray, auto_rotation: bool = True, angle_threshold: float = 5) -> Dict:
        """
        Main correction function.
        
        Args:
            image (np.ndarray): Input image
            auto_rotation (bool): Enable auto rotation detection
            angle_threshold (float): Rotation angle threshold
            
        Returns:
            dict: {'image': corrected_image, 'angle': rotation_angle}
        """
        
        result = image.copy()
        angle = 0
        
        if auto_rotation:
            # Detect rotation angle
            angle = self._detect_rotation(image)
            
            if abs(angle) > angle_threshold:
                self.logger.info(f"📐 Detected rotation: {angle:.2f}°")
                result = self._rotate_image(image, angle)
                self.logger.info(f"✓ Applied rotation correction")
        
        return {
            'image': result,
            'angle': angle
        }
    
    def _detect_rotation(self, image: np.ndarray) -> float:
        """
        Detect rotation angle menggunakan edge detection.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            float: Rotation angle in degrees
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect edges
            edges = cv2.Canny(gray, 50, 150)
            
            # Hough line transform
            lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
            
            if lines is None or len(lines) == 0:
                return 0
            
            # Extract angles
            angles = []
            for line in lines:
                rho, theta = line[0]
                # Convert radian to degree
                angle = np.degrees(theta) - 90
                angles.append(angle)
            
            # Get median angle (robust to outliers)
            median_angle = np.median(angles)
            
            # Normalize angle to [-90, 90]
            if median_angle > 90:
                median_angle -= 180
            elif median_angle < -90:
                median_angle += 180
            
            return float(median_angle)
            
        except Exception as e:
            self.logger.warning(f"⚠️  Rotation detection failed: {e}")
            return 0
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate image dengan padding.
        
        Args:
            image (np.ndarray): Input image
            angle (float): Rotation angle in degrees
            
        Returns:
            np.ndarray: Rotated image
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new size
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # Adjust rotation matrix untuk centering
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]
        
        # Apply rotation
        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (new_w, new_h),
            borderMode=cv2.BORDER_REFLECT_101
        )
        
        return rotated
    
    def manual_rotate(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Manual rotation dengan specified angle.
        
        Args:
            image (np.ndarray): Input image
            angle (float): Rotation angle in degrees
            
        Returns:
            np.ndarray: Rotated image
        """
        return self._rotate_image(image, angle)
    
    def perspective_correction(self, image: np.ndarray, src_points: np.ndarray | None = None, dst_points: np.ndarray | None = None) -> np.ndarray:
        """
        Perspective correction untuk foto yang tertekuk.
        
        Args:
            image (np.ndarray): Input image
            src_points (np.ndarray): Source corner points (4x2)
            dst_points (np.ndarray): Destination corner points (4x2)
            
        Returns:
            np.ndarray: Perspective-corrected image
        """
        
        h, w = image.shape[:2]
        
        # Default: assume rectangle shape
        if src_points is None:
            src_points = np.array([
                [0, 0],
                [w-1, 0],
                [0, h-1],
                [w-1, h-1]
            ], dtype=np.float32)
        else:
            src_points = np.array(src_points, dtype=np.float32)
        
        if dst_points is None:
            dst_points = np.array([
                [0, 0],
                [w-1, 0],
                [0, h-1],
                [w-1, h-1]
            ], dtype=np.float32)
        else:
            dst_points = np.array(dst_points, dtype=np.float32)
        
        # Get perspective transformation matrix
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Apply perspective transformation
        result = cv2.warpPerspective(
            image,
            perspective_matrix,
            (w, h),
            borderMode=cv2.BORDER_REFLECT_101
        )
        
        return result
    
    def auto_crop(self, image: np.ndarray, border_threshold: int = 10) -> np.ndarray:
        """
        Auto crop unnecessary borders dari rotated image.
        
        Args:
            image (np.ndarray): Input image
            border_threshold (int): Threshold untuk border detection
            
        Returns:
            np.ndarray: Cropped image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find non-zero regions
        _, thresh = cv2.threshold(gray, border_threshold, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return image
        
        # Get bounding box dari largest contour
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        
        # Crop with small margin
        margin = 5
        x = max(0, x - margin)
        y = max(0, y - margin)
        
        return image[y:y+h+margin*2, x:x+w+margin*2]