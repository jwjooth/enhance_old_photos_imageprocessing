"""
IMAGE ANALYZER MODULE
Auto-detect image conditions dan recommend enhancement methods

Author: Photo Enhancement Team
Version: 1.0.0
"""

import cv2
import numpy as np
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class ImageAnalyzer:
    """
    Analyze image conditions dan recommend optimal enhancement parameters
    """
    
    def __init__(self):
        """Initialize analyzer"""
        self.logger = logging.getLogger(__name__)
    
    def analyze(self, image: np.ndarray) -> Dict:
        """
        Complete image analysis.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            dict: Analysis results dengan recommendations
        """
        
        analysis = {
            'rotation': self._analyze_rotation(image),
            'noise': self._analyze_noise(image),
            'brightness': self._analyze_brightness(image),
            'contrast': self._analyze_contrast(image),
            'blur': self._analyze_blur(image),
            'color_fading': self._analyze_color_fading(image),
        }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(analysis)
        
        return {
            'analysis': analysis,
            'recommendations': recommendations,
            'report': self._generate_report(analysis, recommendations)
        }
    
    def _analyze_rotation(self, image: np.ndarray) -> Dict:
        """
        Detect if image is rotated.
        
        Returns:
            dict: {'needs_correction': bool, 'angle': float, 'confidence': float}
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
            
            if lines is None or len(lines) == 0:
                return {
                    'needs_correction': False,
                    'angle': 0,
                    'confidence': 0.0,
                    'severity': 'none'
                }
            
            angles = []
            for line in lines:
                rho, theta = line[0]
                angle = np.degrees(theta) - 90
                angles.append(angle)
            
            median_angle = float(np.median(angles))
            
            if median_angle > 90:
                median_angle -= 180
            elif median_angle < -90:
                median_angle += 180
            
            # Confidence berdasarkan absolute angle
            confidence = min(abs(median_angle) / 45.0, 1.0)
            
            # Severity levels
            if abs(median_angle) < 2:
                severity = 'none'
                needs_correction = False
            elif abs(median_angle) < 5:
                severity = 'minor'
                needs_correction = True
            elif abs(median_angle) < 15:
                severity = 'moderate'
                needs_correction = True
            else:
                severity = 'severe'
                needs_correction = True
            
            self.logger.debug(f"Rotation: angle={median_angle:.2f}°, severity={severity}")
            
            return {
                'needs_correction': needs_correction,
                'angle': median_angle,
                'confidence': confidence,
                'severity': severity
            }
            
        except Exception as e:
            self.logger.warning(f"Rotation analysis failed: {e}")
            return {
                'needs_correction': False,
                'angle': 0,
                'confidence': 0.0,
                'severity': 'none'
            }
    
    def _analyze_noise(self, image: np.ndarray) -> Dict:
        """
        Detect noise level menggunakan Laplacian variance.
        
        Returns:
            dict: {'needs_filtering': bool, 'level': float, 'severity': str}
        """
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Calculate Laplacian variance (sharpness indicator)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            variance = laplacian.var()
            
            # Normalize (typical range: 0-1000)
            normalized_variance = min(variance / 1000.0, 1.0)
            
            # Severity levels
            if variance < 100:
                severity = 'severe'
                needs_filtering = True
                method = 'nlm'  # Heavy noise = use NLM
                strength = 1.5
            elif variance < 300:
                severity = 'moderate'
                needs_filtering = True
                method = 'bilateral'
                strength = 1.0
            elif variance < 500:
                severity = 'mild'
                needs_filtering = True
                method = 'bilateral'
                strength = 0.7
            else:
                severity = 'none'
                needs_filtering = False
                method = 'bilateral'
                strength = 0.5
            
            self.logger.debug(f"Noise: variance={variance:.2f}, severity={severity}")
            
            return {
                'needs_filtering': needs_filtering,
                'level': float(normalized_variance),
                'severity': severity,
                'recommended_method': method,
                'recommended_strength': strength
            }
            
        except Exception as e:
            self.logger.warning(f"Noise analysis failed: {e}")
            return {
                'needs_filtering': False,
                'level': 0.0,
                'severity': 'none',
                'recommended_method': 'bilateral',
                'recommended_strength': 0.5
            }
    
    def _analyze_brightness(self, image: np.ndarray) -> Dict:
        """
        Detect brightness level (is image too dark?).
        
        Returns:
            dict: {'needs_enhancement': bool, 'level': float, 'severity': str}
        """
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Calculate average brightness (0-255)
            brightness = np.mean(gray)
            normalized = brightness / 255.0
            
            # Severity levels
            if brightness < 50:
                severity = 'severe'
                needs_enhancement = True
                clip_limit = 3.5
            elif brightness < 100:
                severity = 'moderate'
                needs_enhancement = True
                clip_limit = 2.5
            elif brightness < 150:
                severity = 'mild'
                needs_enhancement = True
                clip_limit = 2.0
            else:
                severity = 'none'
                needs_enhancement = False
                clip_limit = 1.5
            
            self.logger.debug(f"Brightness: {brightness:.2f}, severity={severity}")
            
            return {
                'needs_enhancement': needs_enhancement,
                'level': float(normalized),
                'brightness_value': float(brightness),
                'severity': severity,
                'recommended_clip_limit': clip_limit
            }
            
        except Exception as e:
            self.logger.warning(f"Brightness analysis failed: {e}")
            return {
                'needs_enhancement': False,
                'level': 0.5,
                'brightness_value': 127.5,
                'severity': 'none',
                'recommended_clip_limit': 2.0
            }
    
    def _analyze_contrast(self, image: np.ndarray) -> Dict:
        """
        Detect contrast level.
        
        Returns:
            dict: {'level': float, 'severity': str}
        """
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            contrast = np.std(gray)
            normalized = min(contrast / 127.0, 1.0)  # Max std is ~127
            
            if contrast < 30:
                severity = 'severe'
            elif contrast < 50:
                severity = 'moderate'
            elif contrast < 80:
                severity = 'mild'
            else:
                severity = 'good'
            
            self.logger.debug(f"Contrast: {contrast:.2f}, severity={severity}")
            
            return {
                'level': float(normalized),
                'value': float(contrast),
                'severity': severity
            }
            
        except Exception as e:
            self.logger.warning(f"Contrast analysis failed: {e}")
            return {
                'level': 0.5,
                'value': 64.0,
                'severity': 'mild'
            }
    
    def _analyze_blur(self, image: np.ndarray) -> Dict:
        """
        Detect if image is blurry menggunakan Laplacian variance.
        
        Returns:
            dict: {'is_blurry': bool, 'sharpness': float}
        """
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            variance = laplacian.var()
            
            # Typical threshold: 100
            is_blurry = variance < 100
            normalized = min(variance / 500.0, 1.0)
            
            self.logger.debug(f"Blur: variance={variance:.2f}, is_blurry={is_blurry}")
            
            return {
                'is_blurry': is_blurry,
                'sharpness': float(normalized),
                'variance': float(variance)
            }
            
        except Exception as e:
            self.logger.warning(f"Blur analysis failed: {e}")
            return {
                'is_blurry': False,
                'sharpness': 0.5,
                'variance': 250.0
            }
    
    def _analyze_color_fading(self, image: np.ndarray) -> Dict:
        """
        Detect if colors are faded (saturation level).
        
        Returns:
            dict: {'is_faded': bool, 'saturation': float, 'severity': str}
        """
        try:
            if len(image.shape) != 3:
                return {
                    'is_faded': False,
                    'saturation': 0.5,
                    'severity': 'none'
                }
            
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            s = hsv[:, :, 1]
            
            saturation = np.mean(s) / 255.0
            
            if saturation < 0.3:
                severity = 'severe'
                is_faded = True
            elif saturation < 0.5:
                severity = 'moderate'
                is_faded = True
            else:
                severity = 'none'
                is_faded = False
            
            self.logger.debug(f"Color fading: saturation={saturation:.2f}, severity={severity}")
            
            return {
                'is_faded': is_faded,
                'saturation': float(saturation),
                'severity': severity
            }
            
        except Exception as e:
            self.logger.warning(f"Color fading analysis failed: {e}")
            return {
                'is_faded': False,
                'saturation': 0.5,
                'severity': 'none'
            }
    
    def _generate_recommendations(self, analysis: Dict) -> Dict:
        """
        Generate recommendations based on analysis.
        
        Returns:
            dict: {'geometric', 'filtering', 'histogram', 'overall_notes'}
        """
        
        recommendations = {
            'geometric': {
                'apply': analysis['rotation']['needs_correction'],
                'reason': f"Image rotated {analysis['rotation']['angle']:.1f}° ({analysis['rotation']['severity']})",
                'auto_rotation': True,
                'angle_threshold': 5
            },
            'filtering': {
                'apply': analysis['noise']['needs_filtering'],
                'method': analysis['noise']['recommended_method'],
                'strength': analysis['noise']['recommended_strength'],
                'reason': f"Noise level {analysis['noise']['severity']}",
                'combined': analysis['noise']['severity'] == 'severe'
            },
            'histogram': {
                'apply': analysis['brightness']['needs_enhancement'],
                'clip_limit': analysis['brightness']['recommended_clip_limit'],
                'reason': f"Brightness {analysis['brightness']['severity']} ({analysis['brightness']['brightness_value']:.0f}/255)",
                'color_balance': analysis['color_fading']['is_faded'],
                'local_contrast': analysis['contrast']['severity'] in ['moderate', 'severe']
            },
            'overall_notes': self._generate_notes(analysis)
        }
        
        return recommendations
    
    def _generate_notes(self, analysis: Dict) -> str:
        """Generate human-readable notes."""
        
        notes = []
        
        # Rotation
        if analysis['rotation']['needs_correction']:
            notes.append(f"🔄 Rotation: {analysis['rotation']['angle']:.1f}° ({analysis['rotation']['severity']})")
        
        # Noise
        if analysis['noise']['needs_filtering']:
            notes.append(f"🧹 Noise: {analysis['noise']['severity']} - Using {analysis['noise']['recommended_method']} filter")
        
        # Brightness
        if analysis['brightness']['needs_enhancement']:
            notes.append(f"💡 Brightness: {analysis['brightness']['severity']} ({analysis['brightness']['brightness_value']:.0f}/255)")
        
        # Blur
        if analysis['blur']['is_blurry']:
            notes.append(f"⚠️  Image is blurry (sharpness: {analysis['blur']['variance']:.0f})")
        
        # Color fading
        if analysis['color_fading']['is_faded']:
            notes.append(f"🎨 Color fading: {analysis['color_fading']['severity']} (saturation: {analysis['color_fading']['saturation']:.2f})")
        
        # Contrast
        if analysis['contrast']['severity'] in ['moderate', 'severe']:
            notes.append(f"📊 Contrast: {analysis['contrast']['severity']} (std: {analysis['contrast']['value']:.1f})")
        
        if not notes:
            notes.append("✅ Image is in good condition - minimal enhancement needed")
        
        return " | ".join(notes)
    
    def _generate_report(self, analysis: Dict, recommendations: Dict) -> Dict:
        """Generate detailed report."""
        
        return {
            'metrics': {
                'rotation_angle': analysis['rotation']['angle'],
                'noise_level': analysis['noise']['level'],
                'brightness': analysis['brightness']['brightness_value'],
                'contrast': analysis['contrast']['value'],
                'sharpness': analysis['blur']['sharpness'],
                'color_saturation': analysis['color_fading']['saturation']
            },
            'image_quality': {
                'rotation': analysis['rotation']['severity'],
                'noise': analysis['noise']['severity'],
                'brightness': analysis['brightness']['severity'],
                'blur': 'blurry' if analysis['blur']['is_blurry'] else 'sharp',
                'color': 'faded' if analysis['color_fading']['is_faded'] else 'normal',
                'contrast': analysis['contrast']['severity']
            },
            'enhancements_applied': {
                'geometric': recommendations['geometric']['apply'],
                'filtering': recommendations['filtering']['apply'],
                'histogram': recommendations['histogram']['apply']
            },
            'enhancements_details': recommendations,
            'summary': recommendations['overall_notes']
        }