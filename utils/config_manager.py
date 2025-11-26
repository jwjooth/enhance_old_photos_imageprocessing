import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manage configuration files"""
    
    DEFAULT_CONFIG = {
        "paths": {
            "input": "input",
            "output": "output",
            "models": "models"
        },
        "geometric": {
            "auto_rotation": True,
            "angle_threshold": 5,
            "perspective_correction": False
        },
        "filtering": {
            "method": "bilateral",
            "strength": 1.0,
            "combined_filters": False,
            "scratch_removal": True
        },
        "histogram": {
            "method": "clahe",
            "clip_limit": 2.0,
            "color_balance": True,
            "local_contrast": True
        },
        "processing": {
            "save_intermediate": True,
            "save_comparison": True,
            "quality": 95,
            "max_image_size": 2048
        }
    }
    
    @staticmethod
    def load_config(config_path='config/settings.json'):
        """
        Load configuration dari file.
        
        Args:
            config_path (str): Path ke config file
            
        Returns:
            dict: Configuration
        """
        
        try:
            config_file = Path(config_path)
            
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                logger.info(f"✓ Config loaded: {config_path}")
                return config
            else:
                logger.warning(f"Config not found: {config_path}")
                logger.info("Using default config")
                
                # Create default config
                ConfigManager.save_config(
                    ConfigManager.DEFAULT_CONFIG,
                    config_path
                )
                return ConfigManager.DEFAULT_CONFIG
                
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            logger.info("Using default config")
            return ConfigManager.DEFAULT_CONFIG
    
    @staticmethod
    def save_config(config, config_path='config/settings.json'):
        """
        Save configuration ke file.
        
        Args:
            config (dict): Configuration dict
            config_path (str): Output path
            
        Returns:
            bool: Success or failure
        """
        
        try:
            config_file = Path(config_path)
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"✓ Config saved: {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            return False
    
    @staticmethod
    def get_default_config():
        """Return default config"""
        return ConfigManager.DEFAULT_CONFIG