"""
Pytest Configuration File
Location: tests/conftest.py

Shared fixtures dan configuration untuk semua tests
"""

import pytest
import numpy as np
import cv2
import logging
from pathlib import Path

# Setup logging untuk tests
logging.basicConfig(level=logging.INFO)


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create temporary directory untuk test data"""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture
def caplog_level(caplog):
    """Set log capture level"""
    caplog.set_level(logging.DEBUG)
    return caplog


@pytest.fixture
def benchmark_timer():
    """Simple benchmark timer fixture"""
    import time
    
    class Timer:
        """Timer class untuk benchmark"""
        def __init__(self):
            self.start = None
            self.end = None
        
        def __enter__(self):
            self.start = time.time()
            return self
        
        def __exit__(self, *args):
            self.end = time.time()
        
        @property
        def elapsed(self):
            """Get elapsed time in seconds"""
            if self.start is not None and self.end is not None:
                return self.end - self.start
            return None
    
    return Timer


@pytest.fixture(autouse=True)
def reset_modules():
    """Reset imported modules between tests"""
    yield
    # Cleanup after test


# ============================================================================
# PYTEST HOOKS
# ============================================================================

def pytest_configure(config):
    """Called after command line options have been parsed"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    for item in items:
        # Add markers based on test name
        if "slow" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)