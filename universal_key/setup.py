# -*- coding: utf-8 -*-
"""
Universal Key Setup
==================

Standalone Python package setup for Universal Key.
"""

from setuptools import setup, find_packages
import os

# Read README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="universal-key",
    version="1.0.0",
    author="Cevahir Development Team",
    author_email="dev@cevahir.ai",
    description="🗝️ Universal Key - Cevahir'in Evrensel Yetenek Sistemi",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cevahir-ai/universal-key",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Distributed Computing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.1.0",
            "pytest-asyncio>=0.19.0",
            "pytest-cov>=3.0.0",
            "black>=22.6.0",
            "flake8>=5.0.0",
            "mypy>=0.971"
        ],
        "quantum": [
            "qiskit>=0.39.0",
            "cirq>=1.0.0"
        ],
        "vision": [
            "opencv-python>=4.6.0",
            "Pillow>=9.2.0"
        ],
        "audio": [
            "librosa>=0.9.0",
            "soundfile>=0.10.0"
        ],
        "ml": [
            "torch>=1.12.0",
            "transformers>=4.21.0",
            "scikit-learn>=1.1.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "universal-key=universal_key.uk_main:main",
            "uk=universal_key.uk_main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "universal_key": [
            "*.md",
            "*.txt",
            "*.ini",
            "*/*.md"
        ]
    },
    project_urls={
        "Bug Reports": "https://github.com/cevahir-ai/universal-key/issues",
        "Source": "https://github.com/cevahir-ai/universal-key",
        "Documentation": "https://docs.cevahir.ai/universal-key",
    },
    keywords=[
        "artificial-intelligence", "ai", "machine-learning", "quantum-computing",
        "consciousness", "autonomous-learning", "web-scraping", "robotics",
        "creativity", "security", "temporal-manipulation", "cevahir"
    ],
)
