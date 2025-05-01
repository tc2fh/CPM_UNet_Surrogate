# Surrogate Modeling of Cellular-Potts Agent-Based Models

This repository contains the code to generate training data and perform training of the surrogate models discussed in the paper:

**Surrogate modeling of Cellular-Potts Agent-Based Models as a segmentation task using the U-Net neural network architecture.**

---

## Dependencies

Below are the dependencies required to run the code in this repo, along with links to relevant documentation:

- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)
- [PyTorch](https://pytorch.org/)
- [Albumentations](https://albumentations.ai/)
- [Zarr](https://zarr.dev/)
- [CompuCell3D](https://compucell3d.org/)
- [Matplotlib](https://matplotlib.org/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [SciPy](https://scipy.org/)
- [Scikit-Image](https://scikit-image.org/)

---

## Installation

We recommend installing all dependencies into a conda environment. The CompuCell3D Python API is used to run the simulation software and generate data.

For information on installing Conda and setting up a Conda environment, we recommend the following guide:  
[Getting Started with Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main)