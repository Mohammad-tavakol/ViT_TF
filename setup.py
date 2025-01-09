from setuptools import setup, find_packages

setup(
    name="ViT_TF",
    version="0.1.0",
    description="A package for Vision Transformer models for classification, super resolution and more with tensorflow",
    author="Your Name",
    author_email="your_email@example.com",
    url="https://github.com/yourusername/vit_models",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.9.0",
        "numpy",
        "matplotlib"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)