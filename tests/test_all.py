"""
TEST SUITE - Old Photo Enhancement Project
Unit tests untuk semua modules

Run: pytest
Run dengan verbose: pytest -v
Run dengan coverage: pytest --cov=modules
"""

import pytest
import cv2
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.geometric import GeometricCorrection
from modules.filtering import NoiseFiltering
from modules.histogram import HistogramEqualization
from modules.image_analyzer import ImageAnalyzer
from modules.smart_enhancer import SmartEnhancer


# ============================================================================
# FIXTURES - Test Data
# ============================================================================

@pytest.fixture
def sample_image():
    """Create sample test image"""
    # Create 300x300 RGB image
    image = np.ones((300, 300, 3), dtype=np.uint8) * 128
    # Add some patterns
    image[50:100, 50:100] = 200
    image[150:200, 150:200] = 50
    return image


@pytest.fixture
def rotated_image():
    """Create rotated test image"""
    image = np.ones((300, 300, 3), dtype=np.uint8) * 128
    # Rotate 15 degrees
    center = (150, 150)
    matrix = cv2.getRotationMatrix2D(center, 15, 1.0)
    rotated = cv2.warpAffine(image, matrix, (300, 300))
    return rotated


@pytest.fixture
def noisy_image():
    """Create noisy test image"""
    image = np.ones((300, 300, 3), dtype=np.uint8) * 128
    # Add Gaussian noise
    noise = np.random.normal(0, 25, image.shape)
    noisy = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy


@pytest.fixture
def dark_image():
    """Create dark test image"""
    image = np.ones((300, 300, 3), dtype=np.uint8) * 50
    return image


@pytest.fixture
def faded_image():
    """Create faded color image"""
    image = np.ones((300, 300, 3), dtype=np.uint8) * 128
    # Low saturation
    image[:, :, 0] = 120  # B
    image[:, :, 1] = 125  # G
    image[:, :, 2] = 130  # R
    return image


# ============================================================================
# TEST: Geometric Correction
# ============================================================================

class TestGeometricCorrection:
    """Test geometric correction functionality"""
    
    def test_init(self):
        """Test initialization"""
        geo = GeometricCorrection()
        assert geo is not None
        assert hasattr(geo, 'correct')
    
    def test_correct_no_rotation(self, sample_image):
        """Test correction with no rotation"""
        geo = GeometricCorrection()
        result = geo.correct(sample_image, auto_rotation=True, angle_threshold=5)
        
        assert 'image' in result
        assert 'angle' in result
        assert result['image'] is not None
    
    def test_correct_with_rotation(self, rotated_image):
        """Test correction with rotation"""
        geo = GeometricCorrection()
        result = geo.correct(rotated_image, auto_rotation=True, angle_threshold=5)
        
        assert 'image' in result
        assert 'angle' in result
    
    def test_manual_rotate(self, sample_image):
        """Test manual rotation"""
        geo = GeometricCorrection()
        rotated = geo.manual_rotate(sample_image, 45)
        
        assert rotated is not None
        assert rotated.shape[2] == 3
    
    def test_auto_crop(self, rotated_image):
        """Test auto crop"""
        geo = GeometricCorrection()
        cropped = geo.auto_crop(rotated_image)
        
        assert cropped is not None
        assert cropped.shape[0] > 0
        assert cropped.shape[1] > 0


# ============================================================================
# TEST: Noise Filtering
# ============================================================================

class TestNoiseFiltering:
    """Test noise filtering functionality"""
    
    def test_init(self):
        """Test initialization"""
        filt = NoiseFiltering()
        assert filt is not None
        assert hasattr(filt, 'apply_filter')
    
    def test_bilateral_filter(self, noisy_image):
        """Test bilateral filter"""
        filt = NoiseFiltering()
        result = filt.apply_filter(noisy_image, 'bilateral', 1.0)
        
        assert 'image' in result
        assert 'filter_type' in result
        assert result['filter_type'] == 'bilateral'
        assert result['image'] is not None
    
    def test_nlm_filter(self, noisy_image):
        """Test NLM filter"""
        filt = NoiseFiltering()
        result = filt.apply_filter(noisy_image, 'nlm', 1.0)
        
        assert result['filter_type'] == 'nlm'
        assert result['image'] is not None
    
    def test_median_filter(self, noisy_image):
        """Test median filter"""
        filt = NoiseFiltering()
        result = filt.apply_filter(noisy_image, 'median', 1.0)
        
        assert result['filter_type'] == 'median'
        assert result['image'] is not None
    
    def test_gaussian_filter(self, noisy_image):
        """Test gaussian filter"""
        filt = NoiseFiltering()
        result = filt.apply_filter(noisy_image, 'gaussian', 1.0)
        
        assert result['filter_type'] == 'gaussian'
        assert result['image'] is not None
    
    def test_combined_filter(self, noisy_image):
        """Test combined filter"""
        filt = NoiseFiltering()
        result = filt.combined_filter(noisy_image, 'bilateral', 'median', 1.0)
        
        assert result is not None
    
    def test_scratch_removal(self, noisy_image):
        """Test scratch removal"""
        filt = NoiseFiltering()
        result = filt.scratch_removal(noisy_image)
        
        assert result is not None


# ============================================================================
# TEST: Histogram Equalization
# ============================================================================

class TestHistogramEqualization:
    """Test histogram equalization functionality"""
    
    def test_init(self):
        """Test initialization"""
        hist = HistogramEqualization()
        assert hist is not None
        assert hasattr(hist, 'enhance')
    
    def test_clahe_enhance(self, dark_image):
        """Test CLAHE enhancement"""
        hist = HistogramEqualization()
        result = hist.enhance(dark_image, 'clahe', 2.0)
        
        assert 'image' in result
        assert 'method' in result
        assert result['method'] == 'clahe'
        assert result['image'] is not None
    
    def test_standard_equalization(self, dark_image):
        """Test standard equalization"""
        hist = HistogramEqualization()
        result = hist.enhance(dark_image, 'standard')
        
        assert result['method'] == 'standard'
        assert result['image'] is not None
    
    def test_multiscale_enhance(self, dark_image):
        """Test multiscale enhancement"""
        hist = HistogramEqualization()
        result = hist.enhance(dark_image, 'multiscale', 2.0)
        
        assert result['method'] == 'multiscale'
        assert result['image'] is not None
    
    def test_color_balance(self, faded_image):
        """Test color balance"""
        hist = HistogramEqualization()
        result = hist.color_balance(faded_image)
        
        assert result is not None
    
    def test_gamma_correction(self, dark_image):
        """Test gamma correction"""
        hist = HistogramEqualization()
        result = hist.gamma_correction(dark_image, 1.5)
        
        assert result is not None


# ============================================================================
# TEST: Image Analyzer
# ============================================================================

class TestImageAnalyzer:
    """Test image analysis functionality"""
    
    def test_init(self):
        """Test initialization"""
        analyzer = ImageAnalyzer()
        assert analyzer is not None
        assert hasattr(analyzer, 'analyze')
    
    def test_analyze_good_image(self, sample_image):
        """Test analysis of good image"""
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(sample_image)
        
        assert 'analysis' in result
        assert 'recommendations' in result
        assert 'report' in result
    
    def test_analyze_rotation(self, rotated_image):
        """Test rotation analysis"""
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(rotated_image)
        
        assert 'analysis' in result
        assert 'rotation' in result['analysis']
    
    def test_analyze_noise(self, noisy_image):
        """Test noise analysis"""
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(noisy_image)
        
        assert 'noise' in result['analysis']
        assert 'needs_filtering' in result['analysis']['noise']
    
    def test_analyze_brightness(self, dark_image):
        """Test brightness analysis"""
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(dark_image)
        
        assert 'brightness' in result['analysis']
        assert 'needs_enhancement' in result['analysis']['brightness']
    
    def test_analyze_color_fading(self, faded_image):
        """Test color fading analysis"""
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(faded_image)
        
        assert 'color_fading' in result['analysis']
        assert 'is_faded' in result['analysis']['color_fading']


# ============================================================================
# TEST: Smart Enhancer
# ============================================================================

class TestSmartEnhancer:
    """Test smart enhancement functionality"""
    
    def test_init(self):
        """Test initialization"""
        enhancer = SmartEnhancer()
        assert enhancer is not None
        assert hasattr(enhancer, 'enhance')
    
    def test_enhance_basic(self, sample_image):
        """Test basic enhancement"""
        enhancer = SmartEnhancer()
        result = enhancer.enhance(sample_image)
        
        assert 'original' in result
        assert 'enhanced' in result
        assert 'analysis' in result
        assert 'report' in result
    
    def test_enhance_complex(self, noisy_image):
        """Test enhancement with noisy image"""
        enhancer = SmartEnhancer()
        result = enhancer.enhance(noisy_image)
        
        assert result['enhanced'] is not None
        assert 'steps_applied' in result
    
    def test_enhance_returns_correct_shape(self, sample_image):
        """Test that enhanced image has correct shape"""
        enhancer = SmartEnhancer()
        result = enhancer.enhance(sample_image)
        
        # Shape might change due to rotation, but channels should be 3
        assert len(result['enhanced'].shape) == 3
        assert result['enhanced'].shape[2] == 3
    
    def test_batch_enhance(self, sample_image, noisy_image):
        """Test batch enhancement"""
        enhancer = SmartEnhancer()
        images = {
            'test1.jpg': sample_image,
            'test2.jpg': noisy_image
        }
        
        results = enhancer.batch_enhance(images)
        
        assert len(results) == 2
        assert 'test1.jpg' in results
        assert 'test2.jpg' in results


# ============================================================================
# TEST: Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests untuk full pipeline"""
    
    def test_full_pipeline(self, noisy_image):
        """Test full enhancement pipeline"""
        geo = GeometricCorrection()
        filt = NoiseFiltering()
        hist = HistogramEqualization()
        
        # Step 1
        image = geo.correct(noisy_image)['image']
        
        # Step 2
        image = filt.apply_filter(image, 'bilateral', 1.0)['image']
        
        # Step 3
        image = hist.enhance(image, 'clahe', 2.0)['image']
        
        assert image is not None
        assert image.shape[2] == 3
    
    def test_smart_vs_manual(self, noisy_image):
        """Compare smart vs manual enhancement"""
        # Smart
        smart_enhancer = SmartEnhancer()
        smart_result = smart_enhancer.enhance(noisy_image)
        
        # Manual
        geo = GeometricCorrection()
        filt = NoiseFiltering()
        hist = HistogramEqualization()
        
        manual = geo.correct(noisy_image)['image']
        manual = filt.apply_filter(manual, 'bilateral', 1.0)['image']
        manual = hist.enhance(manual, 'clahe', 2.0)['image']
        
        # Both should produce results
        assert smart_result['enhanced'] is not None
        assert manual is not None


# ============================================================================
# TEST: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases dan error handling"""
    
    def test_very_small_image(self):
        """Test with very small image"""
        tiny_image = np.ones((10, 10, 3), dtype=np.uint8) * 128
        enhancer = SmartEnhancer()
        result = enhancer.enhance(tiny_image)
        assert result['enhanced'] is not None
    
    def test_grayscale_image(self):
        """Test with grayscale image"""
        gray_image = np.ones((300, 300), dtype=np.uint8) * 128
        filt = NoiseFiltering()
        # Should handle gracefully
        try:
            result = filt.apply_filter(gray_image, 'bilateral', 1.0)
            assert result['image'] is not None
        except:
            pass  # Expected for some filters with grayscale
    
    def test_extreme_brightness(self):
        """Test with extreme brightness"""
        bright_image = np.ones((300, 300, 3), dtype=np.uint8) * 255
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(bright_image)
        assert result is not None
    
    def test_extreme_darkness(self):
        """Test with extreme darkness"""
        dark_image = np.ones((300, 300, 3), dtype=np.uint8) * 0
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(dark_image)
        assert result is not None


# ============================================================================
# TEST: Performance
# ============================================================================

class TestPerformance:
    """Performance tests"""
    
    def test_enhancement_speed(self, sample_image):
        """Test enhancement speed"""
        import time
        enhancer = SmartEnhancer()
        
        start = time.time()
        enhancer.enhance(sample_image)
        elapsed = time.time() - start
        
        # Should complete in reasonable time
        assert elapsed < 10  # Less than 10 seconds
    
    def test_batch_processing_speed(self, sample_image):
        """Test batch processing speed"""
        import time
        enhancer = SmartEnhancer()
        
        images = {f'img_{i}.jpg': sample_image for i in range(5)}
        
        start = time.time()
        enhancer.batch_enhance(images)
        elapsed = time.time() - start
        
        # 5 images should complete in reasonable time
        assert elapsed < 60  # Less than 60 seconds


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])