"""
SMART ENHANCEMENT MODULE
Otomatis enhance images based on conditions

Author: Photo Enhancement Team
Version: 1.0.0
"""

import cv2
import numpy as np
import logging
from typing import Dict, Tuple

from modules.geometric import GeometricCorrection
from modules.filtering import NoiseFiltering
from modules.histogram import HistogramEqualization
from modules.image_analyzer import ImageAnalyzer

logger = logging.getLogger(__name__)


class SmartEnhancer:
    """
    Smart enhancement yang otomatis sesuaikan metode berdasarkan kondisi gambar
    """
    
    def __init__(self):
        """Initialize smart enhancer"""
        self.logger = logging.getLogger(__name__)
        self.analyzer = ImageAnalyzer()
        self.geometric = GeometricCorrection()
        self.filtering = NoiseFiltering()
        self.histogram = HistogramEqualization()
    
    def enhance(self, image: np.ndarray) -> Dict:
        """
        Smart enhancement dengan auto-detection.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            dict: {
                'original': original image,
                'enhanced': enhanced image,
                'analysis': analysis results,
                'recommendations': enhancement recommendations,
                'report': detailed report,
                'steps_applied': list of steps applied
            }
        """
        
        self.logger.info("🤖 Starting smart enhancement...")
        
        original_image = image.copy()
        enhanced_image = image.copy()
        steps_applied = []
        
        # Step 1: Analyze image
        self.logger.info("📊 Analyzing image...")
        analysis_result = self.analyzer.analyze(image)
        analysis = analysis_result['analysis']
        recommendations = analysis_result['recommendations']
        report = analysis_result['report']
        
        # Step 2: Apply geometric correction if needed
        if recommendations['geometric']['apply']:
            self.logger.info("🔄 Applying geometric correction...")
            try:
                geo_result = self.geometric.correct(
                    enhanced_image,
                    auto_rotation=recommendations['geometric']['auto_rotation'],
                    angle_threshold=recommendations['geometric']['angle_threshold']
                )
                enhanced_image = geo_result['image']
                steps_applied.append({
                    'step': 'Geometric Correction',
                    'method': 'auto-rotation',
                    'angle': geo_result['angle'],
                    'reason': recommendations['geometric']['reason']
                })
                self.logger.info(f"✓ Geometric correction applied (angle: {geo_result['angle']:.2f}°)")
            except Exception as e:
                self.logger.warning(f"Geometric correction failed: {e}")
        
        # Step 3: Apply filtering if needed
        if recommendations['filtering']['apply']:
            self.logger.info(f"🧹 Applying {recommendations['filtering']['method']} filter...")
            try:
                if recommendations['filtering']['combined']:
                    self.logger.info("Using combined filtering (bilateral + NLM)")
                    enhanced_image = self.filtering.combined_filter(
                        enhanced_image,
                        'bilateral',
                        'nlm',
                        strength=recommendations['filtering']['strength']
                    )
                    filter_method = 'bilateral + NLM (combined)'
                else:
                    filt_result = self.filtering.apply_filter(
                        enhanced_image,
                        filter_type=recommendations['filtering']['method'],
                        strength=recommendations['filtering']['strength']
                    )
                    enhanced_image = filt_result['image']
                    filter_method = recommendations['filtering']['method']
                
                steps_applied.append({
                    'step': 'Noise Filtering',
                    'method': filter_method,
                    'strength': recommendations['filtering']['strength'],
                    'reason': recommendations['filtering']['reason']
                })
                self.logger.info(f"✓ Filtering applied ({filter_method})")
            except Exception as e:
                self.logger.warning(f"Filtering failed: {e}")
        
        # Step 4: Apply histogram enhancement if needed
        if recommendations['histogram']['apply']:
            self.logger.info("💡 Applying histogram enhancement...")
            try:
                hist_result = self.histogram.enhance(
                    enhanced_image,
                    method='clahe',
                    clip_limit=recommendations['histogram']['clip_limit']
                )
                enhanced_image = hist_result['image']
                
                steps_applied.append({
                    'step': 'Histogram Enhancement',
                    'method': 'CLAHE',
                    'clip_limit': recommendations['histogram']['clip_limit'],
                    'reason': recommendations['histogram']['reason']
                })
                
                # Apply color balance if needed
                if recommendations['histogram']['color_balance']:
                    self.logger.info("🎨 Applying color balance...")
                    enhanced_image = self.histogram.color_balance(enhanced_image)
                    steps_applied[-1]['color_balance'] = True
                
                # Apply local contrast if needed
                if recommendations['histogram']['local_contrast']:
                    self.logger.info("📊 Enhancing local contrast...")
                    enhanced_image = self.histogram.local_contrast_enhancement(
                        enhanced_image,
                        strength=1.3
                    )
                    steps_applied[-1]['local_contrast'] = True
                
                self.logger.info("✓ Histogram enhancement applied")
            except Exception as e:
                self.logger.warning(f"Histogram enhancement failed: {e}")
        
        self.logger.info(f"✅ Smart enhancement complete! ({len(steps_applied)} steps applied)")
        
        return {
            'original': original_image,
            'enhanced': enhanced_image,
            'analysis': analysis,
            'recommendations': recommendations,
            'report': report,
            'steps_applied': steps_applied,
            'summary': report['summary']
        }
    
    def batch_enhance(self, images: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        """
        Batch enhancement untuk multiple images.
        
        Args:
            images (dict): {filename: image_array}
            
        Returns:
            dict: Results untuk setiap image
        """
        
        results = {}
        
        for idx, (filename, image) in enumerate(images.items(), 1):
            self.logger.info(f"\n[{idx}/{len(images)}] Processing {filename}")
            result = self.enhance(image)
            results[filename] = result
        
        return results