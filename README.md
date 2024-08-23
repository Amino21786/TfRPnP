# Thesis Formulation Report (TFR)
# Plug-and-play regluarisation techniques (PnP) for inverse problems
This folder consists of Python code used for the TFR report and project involving PnP techniques for inverse problems. Primarily the application of computed tomopgraphy (CT) scanning was used for applications involving image reconstruction of MNIST digits, Shepp-Logan Phantom and natural images. 

PnP algorithms:
- Proximal gradient descent (PGD)
- Fast iterative shrinkage algoirthm (FISTA, variant of PGD)
- Alternating direction method of multiplers (ADMM)

PnP denoisers:
- Total variation (TV)
- Block-matching 3D (BM3D)
- Denoising convolutional neural network (DnCNN)
- Dilated-Residual U-network (DRUNet)
- Gradient-step DRUNet (GS-DRUNet)



## Functionality of the files
There was mainly a use of Jupyter notebooks for fast and easy use of running the respective algorithms and denoisers. Each notebook had an array of functions to fit their purpose and use similar elements across the board (e.g algorithms, forward operators, noise etc)
For the deep denoisers' training weights, \path{deepinv} Pytorch library was used with help from their documentation (Github: https://github.com/deepinv/deepinv
Documentation:https://deepinv.github.io/deepinv/index.html).

Notebook files (.pynb):
-
-
-

Python files (.py):



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
python test.py
```

or using coding editors such as VSC to run the individual files. (In the future there will be a construction of a main.py to run all the necessary models needed for results)

# Acknowledgements
Huge thanks to Dr Yury Korolev and Dr Matthias Ehrhardt for supervising during this project. There were many insightful and helpful discussions, alongside valuable support through the duration of the project. The deep NN denoisers' training weights were used from the \path{deepinv} Python Pytorch library, which greatly helped with implementation of the algorithms and gave inspiration for adapting to my work.
This project was part of my 1st year MRes of statistical applied mathematics at the University of Bath (SAMBa) PhD programme.






























