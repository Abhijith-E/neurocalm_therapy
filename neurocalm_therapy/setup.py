"""
Setup script for NeuroCalm Voice Therapy System
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="neurocalm-therapy",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Speech-Based Therapeutic System for Stress Relief and Headache Management",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/neurocalm-therapy",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "librosa>=0.9.2",
        "soundfile>=0.11.0",
        "matplotlib>=3.4.3",
        "scikit-learn>=0.24.2",
        "pydub>=0.25.1",
        "PyAudio>=0.2.11",
        "hmmlearn>=0.2.7",
        "dtaidistance>=2.3.9",
        "python-speech-features>=0.6",
        "tqdm>=4.62.0",
        "pandas>=1.3.0",
        "seaborn>=0.11.2",
        "joblib>=1.0.1",
    ],
    entry_points={
        "console_scripts": [
            "neurocalm=main:demo_complete_system",
        ],
    },
)