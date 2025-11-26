"""
OLD PHOTO ENHANCEMENT - Modules Package

Main modules untuk photo enhancement:
- geometric: Geometric correction
- filtering: Noise filtering
- histogram: Histogram equalization
"""

from .geometric import GeometricCorrection
from .filtering import NoiseFiltering
from .histogram import HistogramEqualization

__version__ = "1.0.0"
__author__ = "Photo Enhancement Team"
__all__ = ['GeometricCorrection', 'NoiseFiltering', 'HistogramEqualization']