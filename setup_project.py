"""
Auto-setup script untuk create folder structure
Jalankan: python setup_project.py
"""

import os
import json
from pathlib import Path


def setup_project():
    """Create project structure"""
    
    print("🔧 Setting up Old Photo Enhancement Project...\n")
    
    # Folders to create
    folders = [
        'input',
        'output/geometric',
        'output/filtered',
        'output/histogram',
        'output/final',
        'output/comparisons',
        'config',
        'logs',
        'modules',
        'utils',
        'docs'
    ]
    
    # Create folders
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {folder}/")
    
    # Default config
    default_config = {
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
    
    # Create config file
    config_path = Path('config/settings.json')
    if not config_path.exists():
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        print(f"✓ Created: config/settings.json")
    
    # Create README in input folder
    input_readme = Path('input/README.md')
    if not input_readme.exists():
        with open(input_readme, 'w') as f:
            f.write("""# Input Folder

Put your old photos here!

Supported formats:
- .jpg / .jpeg
- .png
- .bmp
- .tiff

Example:
- nenek_1970.jpg
- family_photo.png
- old_picture.tiff
""")
        print(f"✓ Created: input/README.md")
    
    print("\n✅ Project setup completed!")
    print("\n📋 Next steps:")
    print("1. Copy old photos to: input/")
    print("2. Run: python main.py")
    print("3. Check results in: output/")


if __name__ == "__main__":
    setup_project()