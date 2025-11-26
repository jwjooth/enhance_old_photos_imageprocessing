"""
OLD PHOTO ENHANCEMENT PROJECT
Main execution file - Orchestrates entire restoration pipeline

Author: Photo Enhancement Team
Version: 1.0.0
Date: 2025
"""

import os
import cv2
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json

# Import modules
from modules.geometric import GeometricCorrection
from modules.filtering import NoiseFiltering
from modules.histogram import HistogramEqualization
from utils.image_loader import ImageLoader
from utils.config_manager import ConfigManager
from utils.logger import setup_logger

# ============================================================================
# SETUP & CONFIGURATION
# ============================================================================

# Setup logger
logger = setup_logger(__name__)

# Load configuration
config = ConfigManager.load_config('config/settings.json')

# ============================================================================
# MAIN ENHANCEMENT CLASS
# ============================================================================

class OldPhotoEnhancement:
    """
    Main orchestrator untuk old photo enhancement pipeline.
    
    Pipeline:
    1. Load image
    2. Geometric correction (jika perlu)
    3. Noise filtering
    4. Histogram equalization
    5. Save results
    """
    
    def __init__(self, config_file='config/settings.json'):
        """Initialize dengan configuration"""
        self.config = ConfigManager.load_config(config_file)
        self.logger = logging.getLogger(__name__)
        
        # Initialize modules
        self.geometric = GeometricCorrection()
        self.filtering = NoiseFiltering()
        self.histogram = HistogramEqualization()
        self.loader = ImageLoader()
        
        # Setup output directories
        self._setup_output_dirs()
        
        self.logger.info("✅ OldPhotoEnhancement initialized")
    
    def _setup_output_dirs(self):
        """Create output directories jika belum exist"""
        output_base = Path(self.config['paths']['output'])
        output_base.mkdir(parents=True, exist_ok=True)
        
        # Sub-folders
        (output_base / 'geometric').mkdir(exist_ok=True)
        (output_base / 'filtered').mkdir(exist_ok=True)
        (output_base / 'histogram').mkdir(exist_ok=True)
        (output_base / 'final').mkdir(exist_ok=True)
        
        self.logger.info("📁 Output directories created")
    
    def enhance_single_image(self, image_path, save_intermediate=True):
        """
        Enhance single image melalui full pipeline.
        
        Args:
            image_path (str): Path ke image
            save_intermediate (bool): Save hasil setiap step
            
        Returns:
            dict: Results dengan semua output
        """
        self.logger.info(f"🖼️  Processing: {image_path}")
        
        try:
            # Step 1: Load image
            image = self.loader.load_image(image_path)
            if image is None:
                self.logger.error(f"❌ Failed to load: {image_path}")
                return None
            
            self.logger.info("✓ Image loaded successfully")
            
            # Step 2: Geometric Correction (Optional)
            self.logger.info("🔄 Step 1/3: Geometric Correction...")
            geometric_result = self.geometric.correct(
                image,
                auto_rotation=self.config['geometric']['auto_rotation'],
                angle_threshold=self.config['geometric']['angle_threshold']
            )
            image_corrected = geometric_result['image']
            
            if save_intermediate:
                self._save_image(
                    geometric_result['image'],
                    image_path,
                    'geometric'
                )
            
            self.logger.info("✓ Geometric correction completed")
            
            # Step 3: Filtering (Noise Removal)
            self.logger.info("🧹 Step 2/3: Noise Filtering...")
            filtering_result = self.filtering.apply_filter(
                image_corrected,
                filter_type=self.config['filtering']['method'],
                strength=self.config['filtering']['strength']
            )
            image_filtered = filtering_result['image']
            
            if save_intermediate:
                self._save_image(
                    image_filtered,
                    image_path,
                    'filtered'
                )
            
            self.logger.info("✓ Filtering completed")
            
            # Step 4: Histogram Equalization (Contrast Enhancement)
            self.logger.info("💡 Step 3/3: Histogram Equalization...")
            histogram_result = self.histogram.enhance(
                image_filtered,
                method=self.config['histogram']['method'],
                clip_limit=self.config['histogram']['clip_limit']
            )
            image_final = histogram_result['image']
            
            if save_intermediate:
                self._save_image(
                    image_final,
                    image_path,
                    'final'
                )
            
            self.logger.info("✓ Histogram equalization completed")
            
            # Compile results
            results = {
                'original': image,
                'geometric': image_corrected,
                'filtered': image_filtered,
                'final': image_final,
                'metadata': {
                    'input_file': image_path,
                    'timestamp': datetime.now().isoformat(),
                    'config': self.config
                }
            }
            
            self.logger.info("✅ Processing completed successfully!\n")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error processing {image_path}: {str(e)}")
            return None
    
    def enhance_batch(self, input_folder):
        """
        Enhance multiple images dalam folder.
        
        Args:
            input_folder (str): Folder dengan images
            
        Returns:
            dict: Results untuk semua images
        """
        self.logger.info(f"📂 Processing batch from: {input_folder}")
        
        input_path = Path(input_folder)
        image_files = list(input_path.glob('*.jpg')) + \
                     list(input_path.glob('*.png')) + \
                     list(input_path.glob('*.jpeg'))
        
        self.logger.info(f"Found {len(image_files)} images to process")
        
        results = {}
        for idx, image_file in enumerate(image_files, 1):
            self.logger.info(f"\n[{idx}/{len(image_files)}] Processing {image_file.name}")
            result = self.enhance_single_image(str(image_file))
            if result:
                results[image_file.name] = result
        
        self.logger.info(f"\n✅ Batch processing completed! ({len(results)}/{len(image_files)} succeeded)")
        return results
    
    def _save_image(self, image, original_path, folder_suffix):
        """Save image ke output folder"""
        output_base = Path(self.config['paths']['output'])
        output_dir = output_base / folder_suffix
        
        filename = Path(original_path).stem
        output_path = output_dir / f"{filename}_enhanced.jpg"
        
        cv2.imwrite(str(output_path), image)
        self.logger.debug(f"💾 Saved: {output_path}")
    
    def create_comparison(self, results, image_name):
        """
        Create side-by-side comparison image.
        
        Args:
            results (dict): Results dari enhancement
            image_name (str): Nama image
            
        Returns:
            np.ndarray: Comparison image
        """
        original = results['original']
        final = results['final']
        
        # Resize if different sizes
        h_orig, w_orig = original.shape[:2]
        h_final, w_final = final.shape[:2]
        
        if (h_orig, w_orig) != (h_final, w_final):
            final = cv2.resize(final, (w_orig, h_orig))
        
        # Create comparison
        comparison = np.hstack([original, final])
        
        # Add labels
        cv2.putText(comparison, "ORIGINAL", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, "ENHANCED", (w_orig + 20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return comparison
    
    def save_comparison(self, results, image_name):
        """Save comparison image"""
        comparison = self.create_comparison(results, image_name)
        
        output_base = Path(self.config['paths']['output'])
        output_path = output_base / f"{Path(image_name).stem}_comparison.jpg"
        
        cv2.imwrite(str(output_path), comparison)
        self.logger.info(f"📊 Comparison saved: {output_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point"""
    
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║         OLD PHOTO ENHANCEMENT PROJECT v1.0.0          ║
    ║                                                       ║
    ║    Restore and enhance your precious old photos!     ║
    ╚═══════════════════════════════════════════════════════╝
    """)
    
    # Initialize enhancement engine
    enhancer = OldPhotoEnhancement('config/settings.json')
    
    # Mode selection
    print("\n🎯 Select Mode:")
    print("1. Single Image Enhancement")
    print("2. Batch Processing")
    
    mode = input("\nEnter choice (1 or 2): ").strip()
    
    if mode == '1':
        # Single image
        image_path = input("Enter image path: ").strip()
        
        if not os.path.exists(image_path):
            print(f"❌ File not found: {image_path}")
            return
        
        results = enhancer.enhance_single_image(image_path)
        
        if results:
            # Create comparison
            enhancer.save_comparison(results, image_path)
            print(f"\n✅ Enhancement completed!")
            print(f"📁 Output saved to: {enhancer.config['paths']['output']}")
    
    elif mode == '2':
        # Batch processing
        folder_path = input("Enter folder path: ").strip()
        
        if not os.path.isdir(folder_path):
            print(f"❌ Folder not found: {folder_path}")
            return
        
        results = enhancer.enhance_batch(folder_path)
        print(f"\n✅ Batch processing completed!")
        print(f"📁 Output saved to: {enhancer.config['paths']['output']}")
    
    else:
        print("❌ Invalid choice!")


if __name__ == "__main__":
    main()