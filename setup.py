from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="old-photo-enhancement",
    version="1.0.0",
    author="Photo Enhancement Team",
    author_email="team@example.com",
    description="AI-powered old photo restoration system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/old-photo-enhancement",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Graphics :: Graphics Conversion",
    ],
    python_requires=">=3.11",
    install_requires=[
        "opencv-python>=4.8.0",
        "numpy>=1.26.0",
        "pillow>=10.0.0",
        "scikit-image>=0.21.0",
        "streamlit>=1.28.0",
    ],
)