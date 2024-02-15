# GeNSIT: Generating Neural Spatial Interaction Tables

**Author:** [Ioannis Zachos](https://yannisza.github.io/)

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org) ![Python](https://img.shields.io/badge/Python-3.9-blue)

# Table of Contents

- [GeNSIT: Generating Neural Spatial Interaction Tables](#gensit-generating-neural-spatial-interaction-tables)
- [Table of Contents](#table-of-contents)
- [Example](#example)
- [Installation guide](#installation-guide)
  - [Docker](#docker)
  - [OSX](#osx)
- [Data](#data)
- [Run experiments](#run-experiments)
  - [Produce plots](#produce-plots)
- [Related publications](#related-publications)
- [Acknowledgments](#acknowledgments)

# Example

# Installation guide

Assuming Python >=3.9.7, (mini)conda and git are installed, clone this repository by running

```
git clone git@github.com:YannisZa/GenSIT.git
```

Once available locally, navigate to the main folder as follows:

```
cd GenSIT
```

## Docker

This section assumes `Docker` has been installed on your machine. Please follow [this guide](https://docs.docker.com/engine/install/) if you wish to install `Docker`. We recommended running `GenSIT` on a Docker container if you do not plan to make any code changes.

Build the docker image image

```
docker build -t "gensit" .
```

Once installed, make sure everything is working by running

```
docker run gensit --help
```

## OSX

If you are using conda do:

```
conda create -y -n gensit python=3.9.7
conda install --force-reinstall -y -q --name gensit -c conda-forge -c pytorch --file requirements.txt
conda install -y conda-build
conda activate gensit
python3 setup.py develop
```

Otherwise, make sure you install the `gensit` command line tool and its dependencies by running

```
pip3 install -e .
```

You can ensure that the dependencies have been successfully installed by running:

```
gensit --help
```

You should get 

# Data

QS702EW - Distance travelled to work

# Run experiments

## Produce plots

# Related publications

- Ioannis Zachos, Mark Girolami, Theodoros Damoulas. _Generating Origin-Destination Matrices in Neural Spatial Interaction Models_. (Under review).
- Ioannis Zachos, Theodoros Damoulas, Mark Girolami. _Table Inference for Combinatorial Origin-Destination Choices in Agent-based Population Synthesis_. [https://arxiv.org/abs/2307.02184](https://arxiv.org/abs/2307.02184) (Stat, 2024).

# Acknowledgments

We acknowledge support from the [UK Research and Innovation (UKRI) Research Council](https://www.ukri.org/), [Arup](https://www.arup.com/) and Cambridge University's [Center for Doctoral Training in Future Infrastructure and Built Environment](https://www.fibe-cdt.eng.cam.ac.uk/). We thank [Arup's City Modelling Lab](https://www.arup.com/services/digital/city-modelling-lab) for their insightful discussions and feedback without which this repository would not have come to fruition.
