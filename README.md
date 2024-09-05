# Thesis Formulation Report (TFR)
# Plug-and-play regularisation techniques (PnP) for inverse problems
This folder consists of Python code used for the TFR report and project involving PnP techniques for inverse problems. Primarily the application of computed tomopgraphy (CT) scanning was used for applications involving image reconstruction of MNIST digits, Shepp-Logan Phantom and natural images. 

PnP algorithms:
- Proximal gradient descent (PGD)
- Fast iterative shrinkage algoirthm (FISTA, variant of PGD)
- Alternating direction method of multiplers (ADMM)

PnP denoisers:
- Total variation (TV) - taken from adapted skimage library for torch tensors (in the utils folder as torch_denoise_tv_chambolle)
- Block-matching 3D (BM3D) - taken from bm3d library
- Denoising convolutional neural network (DnCNN) - taken from deepinv library
- Dilated-Residual U-network (DRUNet) - taken from deepinv library
- Gradient-step DRUNet (GS-DRUNet) - - taken from deepinv library


Other subfolders:
- data and Natural_Images - contains example images including MNIST and butterfly.png (butterfly in my childhood home garden) for application use for CT and general image denoising
- plots - contains results from application of the Python files below that were used in the report

## Functionality of the files
There was mainly a use of Jupyter notebooks for fast and easy use of running the respective algorithms and denoisers. Each notebook had an array of functions to fit their purpose and use similar elements across the board (e.g algorithms, forward operators, noise etc)
For the deep denoisers' training weights, deepinv Pytorch library was used with help from their documentation (Github: https://github.com/deepinv/deepinv
Documentation:https://deepinv.github.io/deepinv/index.html). Note that all files are implemented to work with torch Python library (inputs as torch tensors, can be adapted to numpy as well).

Notebook files (.ipynb):
- MNISTCT - PnP-PGD and PnP-ADMM implementation across the five denoisers for MNIST digits (Section 4.1.1 results)
- PhantomCT - PnP-PGD and PnP-ADMM implementation across the five denoisers for Shepp-Logan Phantom (Section 4.1.2 results) - early version of Section 4.2 results as well for Residual noise PDFs
- NaturalCT - PnP-PGD and PnP-ADMM implementation across the five denoisers for Natural image [Butterfly in my garden :)] - Trial file

Python files (.py):
- plotting.py - functions to produce the plots between the five denoisers and the PSNR and fixed-point convergence vs number of iterations plots used in the above .ipynb files
- algorithms.py - PnP-PGD, PnP-FISTA, PnP-ADMM algorithms with the denoiser choice function used in the above .ipynb files 

## Libraries
In this project, a number of Python libraries were used for model construction, mathematical calculations, plotting and data manipulation (including the standard numpy, matplotlib etc) 
The non-trivial extensively used Python libraries include:
- deepinv -> downloading neural networks (NNs) training weights for the deep denoisers including DnCNN, DRUNet and GSDRUNet (Training weights obtained via https://huggingface.co/deepinv)
- torch and torchvision -> algorithm construction use and use of MNIST digit dataset for application of the algorithms
- skimage --> Shepp-Logan Phantom images and use of Radon transform and filtered back-projection (FBP)
- statsmodels --> alongside numpy and matplotlib, helpful for probability density function plots of noise distributions and histograms plots as well.

Full list of libraries used are stated in the dependencies.yml file.

## Interface instructions
To create the conda environment one can install it via a dependencies file command:
```
conda env create -f dependencies.yml
```
This exports the relevant libraries into a new environment called my_env_project, which can then be activated by:
```
conda activate my_env_project
```

Then, the code can be run through the command line for example
```
python algorithms.py
```

or using coding editors such as VSC to run the individual files. (In the future there will be a construction of a main.py to run all the necessary models needed for results)

# Acknowledgements
Huge thanks to Dr Yury Korolev and Dr Matthias Ehrhardt for supervising during this project. There were many insightful and helpful discussions, alongside valuable support through the duration of the project. The deep NN denoisers' training weights were used from the deepinv Python Pytorch library, which greatly helped with implementation of the algorithms and gave inspiration for adapting to my work.
This project was part of my 1st year MRes of statistical applied mathematics at the University of Bath (SAMBa) PhD programme.






























