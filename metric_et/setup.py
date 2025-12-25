"""Setup script for METRIC ETa package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="metric_et",
    version="0.1.0",
    author="METRIC ETa Developers",
    description="METRIC Evapotranspiration model for Landsat data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/metric_et",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "xarray>=0.19.0",
        "rasterio>=1.2.0",
        "rioxarray>=0.8.0",
        "geopandas>=0.10.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "click>=8.0.0",
        "pyyaml>=6.0",
        "loguru>=0.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=21.0",
            "flake8>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "metric=metric_et.cli.interface:cli",
        ],
    },
    include_package_data=True,
)
