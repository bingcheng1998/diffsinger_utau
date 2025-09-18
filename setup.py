#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

# 读取 README 文件
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# 读取版本信息
with open("voice_bank/__init__.py", "r", encoding="utf-8") as fh:
    for line in fh:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"').strip("'")
            break
    else:
        version = "0.1.0"

setup(
    name="diffsinger-utau",
    version=version,
    author="Your Name",
    author_email="your.email@example.com",
    description="DiffSinger UTAU inference toolkit with CLI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/diffsinger-utau",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8,<3.9",
    install_requires=[
        "numpy>=1.21,<1.25",
        "librosa>=0.9,<0.10",
        "PyYAML>=6.0",
        "onnxruntime>=1.12,<1.17",
        "torch>=1.10,<2.0",
        "pypinyin>=0.40",
        "scipy>=1.7",
    ],
    extras_require={
        "viz": ["matplotlib>=3.5"],
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.900",
        ],
    },
    entry_points={
        "console_scripts": [
            "dsutau=voice_bank.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
