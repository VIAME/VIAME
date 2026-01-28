# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name="darknet-to-pytorch-onnx",  # Required
    version="0.1.0",  # Required
    description="A minimal PyTorch implementation of YOLOv4 with ONNX exporter",  # Optional
    url="https://gitlab.kitware.com/keu-computervision/ml/darknet-to-pytorch-onnx",  # Optional
    packages=find_packages('.'),  # Optional
    python_requires=">=3.7, <4",
    install_requires=[
        "numpy",
        "torch>=1.6.0"],  # Optional
)
