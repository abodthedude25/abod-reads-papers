from setuptools import setup, find_packages

setup(
    name="stream_diffusion",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "diffusers>=0.25.0",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "pillow>=9.5.0",
        "numpy>=1.24.0",
    ],
    python_requires=">=3.8",
)