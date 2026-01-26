from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tiny-language-model",
    version="0.1.0",
    author="Zilong Liu",
    description="Train small language models entirely from scratch using native PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lyuzlion/Tiny-Language-Model",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
    ],
)
