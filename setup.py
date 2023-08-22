from setuptools import setup

setup(
    name = "DDPM-mRNA-augmentation",
    py_modules = ["gaussian_diffusion"],
    install_requires=["blobfile>=1.0.5", "torch", "tqdm"],
)