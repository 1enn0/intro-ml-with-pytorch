# Introduction to Machine Learning with PyTorch

This repository contains some tutorials on machine learning using JupyterLab
notebook and PyTorch.

## Contents of this Repository

* `inference-cpp`: code for the c++ inference application
* `notebooks`: Jupyter notebooks containing the tutorials (including code)
    - `0_intro_machine_learning.ipynb`: Basic introduction to machine learning
    - `1_pytorch_basic.ipynb`: Train a simple neural network to learn a line equation
    - `2_pytorch_advanced.ipynb`: Train an audio classifier model using transfer learning
* `train`: default training directory (used to saved trained model and spectrogram config)
* `test/data`: directory containing some audio files used for testing

## Setup

First, you neeed to clone this repository. Make sure to include the submodule(s) when cloning by running

```
git clone --recurse-submodules https://stash.steinberg-intra.net/scm/~lhannink/intro-ml-with-pytorch.git
```

In case you did a bare `clone` without the submodule flag, run
```
git submodule sync
git submodule update --init --recursive
```

to download the submodules for an existing checkout.

## Dependencies

### General

In order to run the tutorial notebooks and build the C++ application, you need to have the following stuff installed on your system:

* ``cmake`` + supported build system for your platform (Ninja, Make, VS, Xcode, ...)
* Python 3.8
* `pipenv`

If you don't have Python 3.8 and `pipenv` installed have a look at [INSTALLING_PYTHON.md](INSTALLING_PYTHON.md).

### Create Virtual Environment

The tutorial notebooks are intended to be run from within a Python virtual
environment. 

Open up a terminal **in the directory of this repository** and run

```
pipenv install
```

to automatically create a new virtual environment using the correct
Python version and install all package dependencies.

### Start JupyterLab

Start the JupyterLab server inside of your new virtual environment by running

```
pipenv run jupyter lab
```
still inside the directory of this repo.

A new instance of the Juypter server will be started and automatically open
up the JupyterLab start page in your browser. We will only be interacting
with the browser front-end from now on but you need to leave the server
running in the background.

Using the file browser in the left pane, navigate into the `notebooks/`
directory and open up the first `.ipynb` file.

## Test Sounds

This repository contains some sounds from `freesound.org` for testing purposes. The following sounds have been used:
* https://freesound.org/people/AurelioSons/sounds/207457/
* https://freesound.org/people/InspectorJ/sounds/368804/
* https://freesound.org/people/InspectorJ/sounds/404329/
* https://freesound.org/people/Juan_Merie_Venter/sounds/327666/
* https://freesound.org/people/zachrau/sounds/362283/
* https://freesound.org/people/Teumova/sounds/439667/
* https://freesound.org/people/Masgame/sounds/347544/
* https://freesound.org/people/Kingrock2009/sounds/544376/
* https://freesound.org/people/Acekat13/sounds/515685/
* https://freesound.org/people/pryght%20one/sounds/27130/
* https://freesound.org/people/Erokia/sounds/550708/