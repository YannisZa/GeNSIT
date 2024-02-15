# GeNSIT: Generating Neural Spatial Interaction Tables

**Author:** [Ioannis Zachos](https://yannisza.github.io/)

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org) ![Python](https://img.shields.io/badge/Python-3.9-blue)

# Table of Contents

- [GeNSIT: Generating Neural Spatial Interaction Tables](#gensit-generating-neural-spatial-interaction-tables)
- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
  - [Motivation](#motivation)
  - [Contribution](#contribution)
- [Installation](#installation)
  - [Docker](#docker)
  - [OSX](#osx)
  - [Validate installation](#validate-installation)
- [Inputs](#inputs)
  - [Data](#data)
    - [Real-world](#real-world)
    - [Synthetic](#synthetic)
  - [Configs](#configs)
- [Functionality](#functionality)
  - [Run](#run)
  - [Plot](#plot)
  - [Summarise](#summarise)
  - [Reproduce](#reproduce)
- [Related publications](#related-publications)
- [Acknowledgments](#acknowledgments)

> **_Quick Start:_** We recommended going through sections on [Installation](#installation) and [Run](#run) if you wish to run `GeNSIT` using default settings.

# Introduction

<img src="./gensit_framework.jpg" alt="framework" width="1200"/>

## Motivation

Agent-based models (ABMs) have emerged as a pivotal tool for policy-making across various fields, such as transportation, economics, and epidemiology, particularly in response to complex challenges such as the COVID-19 pandemic. ABMs simulate individual agent interactions governed by stochastic dynamical systems, giving rise to an aggregate emergent structure. This allows decision-makers to test different policy interventions under varying conditions. To achieve this effectively, agent populations and their characteristics are synthesized subject to partially observable and fragmented census data at coarse spatio-temporal resolutions.

This _population synthesis_ process aims to generate representative agents whose socio-economic profiles, once spatially aggregated and/or marginalized by agent trait, mirror the observed population-level data. This problem is closely related to estimating joint agent profile proportions (continuous setting) and counts (discrete setting) in _ecological inference_. Agent profiles mainly comprise of discrete categorical variables and the goal is to sample from the joint distribution of these variables. Samples from this distribution are summarised in multi-way contingency tables whose marginal summary statistics or their spatially cruder versions are observed.

A facet of this sampling problem that is pertinent to population synthesis for many types of ABMs, including transportation ABMs, is the synthesis of the discrete spatial locations of agent activities. This is because network-wide simulated travel patterns are predominantly governed by the agent activity locations, which are fixed before simulation. Samples from the discrete joint distribution of trips between origins and destinations are summarised in **origin-destination matrices** (ODMs), namely two-way contingency tables. Observed realisations of this distribution are scarce, especially under refinement of the spatial resolution of locations (i.e. contingency table dimensions). Therefore, aggregate agent activity surveys by geographical region are leveraged. The discrepancy between the data and latent spatial resolutions induces a discrete combinatorial location choice space. Hence, a unique set of individual agent choices consistent with the data cannot be recovered. This unidentifiability issue can be mitigated by exploring this constrained space in a probabilistic manner. Given $M$ agents with $P$ possible origin-destination pairs, there are $P^M$ possible location configurations that need to be integrated over to compute a data likelihood, many of which are inconsistent with the data.

## Contribution

This repository introduces a [computational framework named `GeNSIT`](#introduction) see for exploring the constrained discrete origin-destination matrices of agent trip location choices using closed-form or Gibbs Markov Basis sampling. The underlying continuous choice probability or intensity function (unnormalised probability function) is modelled by total and singly constrained **spatial interaction models** (SIMs) or _gravity models_ embedded in the well-known Harris Wilson stochastic differential equations (SDEs). We employ Neural Networks to calibrate the SIM parameters. We include Markov Chain Monte Carlo (MCMC) schemes leveraged to learn the SIM parameters in previous works. For more details on the mathematical aspects of this repository please look at the [Publications section](#related-publications).

The `GeNSIT` package provides functionality for five different operations: [`create`](#synthetic), [`run`](#run), [`plot`](#plot), [`reproduce`](#reproduce), [`summarise`](#summarise).

# Installation

Assuming Python >=3.9.7 and git are installed, clone this repository by running

```
git clone git@github.com:YannisZa/GeNSIT.git
```

Once available locally, navigate to the main folder as follows:

```
cd GeNSIT
```

> **_Tip:_** We recommended running `GeNSIT` on a `Docker` container if you do not plan to make any code changes.

## Docker

This section assumes `Docker` has been installed on your machine. Please follow [this guide](https://docs.docker.com/engine/install/) if you wish to install `Docker`.
Build the docker image image

```
docker build -t "gensit" .
```

Once installed, make sure everything is working by running

```
docker run gensit --help
```

## OSX

This section assumes `anaconda` or `miniconda` has been installed on your machine. Please follow [this](https://docs.anaconda.com/free/anaconda/install/index.html) or [this](https://docs.anaconda.com/free/miniconda/miniconda-install/) guide if you wish to install either of them. Then, run:

```
conda create -y -n gensit python=3.9.7
conda activate gensit
conda install -y -c conda-forge --file requirements.txt
conda install -y conda-build
python3 setup.py develop
```

Otherwise, make sure you install the `gensit` command line tool and its dependencies by running

```
pip3 install -e .
```

## Validate installation

You can ensure that the dependencies have been successfully installed by running:

```
gensit --help
```

You should get a print statement like this:

```
Usage: gensit [OPTIONS] COMMAND [ARGS]...

  Command line tool for Generating Neural Spatial Interaction Tables (origin-
  destination matrices)

Options:
  --help  Show this message and exit.

Commands:
  create     Create synthetic data for spatial interaction table and...
  plot       Plot experimental outputs.
  reproduce  Reproduce figures in the paper.
  run        Sample discrete spatial interaction tables...
  summarise  Create tabular summary of metadata, metrics computed for...
```

Throughout the remainder of this readme we illustrate `GeNSIT's` command line tool capabilities assuming that a `docker` container has been installed.

# Inputs

Inputs to `GeNSIT` are [**data**](#data) and [**configuration**](#configs) files.

## Data

The minimum data requirements include:

- A set of origin and destination locations between which agents travel.
- A cost matrix $\mathbf{C}$ reflecting inconvenience of travel from any origin to any destination. This can be distance and/or time dependent (e.g. Euclidean distance and/or travel times).
- A measure of destination attractiveness $\mathbf{z}$. This depends on the types of trips agents make e.g. for work trips this would be number of jobs available at each destination.
- The total number of agents/trips $M$. Each agent performs exactly one trip.

Optional datasets may be:

- Origin and/or destination demand.
- Partially observed trips between selected origin-destination pairs.
- Total distance and/or time agents have travelled by origin and/or destination location.
- A transportation network/graph.
- A ground truth agent trip table to validate your model.

### Real-world

We consider agent trips from residence to workplace locations in Cambridge, UK. We use the following datasets from the Census 2011 data provided by the [Office of National Statistics](https://www.ons.gov.uk):

- [Lower super output areas (LSOAs), Middle super output areas (MSOAs)](../data/inputs/cambridge_work_commuter_lsoas_to_msoas/lsoas_to_msoas.geojson) as origin, destination locations, respectively.
- [Average shortest path in a transportation network](../data/inputs/cambridge_work_commuter_lsoas_to_msoas/cost_matrices/clustered_facilities_sample_20x20_20_01_2023_sample_20x20_clustered_facilities_ripleys_k_500_euclidean_points%_prob_origin_destination_adjusted_normalised_boundary_only_edge_corrected_cost_matrix_max_normalised.txt) between a random sample of 20 residences inside each LSOA and 20 workplaces inside each MSOA as a cost matrix.
- [Number of jobs available at each MSOA](./data/inputs/cambridge_work_commuter_lsoas_to_msoas/destination_attraction_time_series_sum_normalised.txt) as a destination attraction proxy used in the NN's loss function.
- [Total distance travelled to work from each LSOA](./data/inputs/cambridge_work_commuter_lsoas_to_msoas/lsoas_total_distance_to_work.txt) as an input to the NN's loss function.
- [Ground truth agent trip table](./data/inputs/cambridge_work_commuter_lsoas_to_msoas/table_lsoas_to_msoas.txt) a validation dataset. Parts of this table such as origin/destination demand (row/colsums) and a random subset of trips (cells) are also conditioned upon acting as table constraint data.

We note the transportation network as well as the residence and workplace locations were extracted using Arup's [`genet`](https://github.com/arup-group/genet) and [`osmox`](https://github.com/arup-group/osmox), respectively. The geo-referenced map used as an input to these tools was downloaded from [Open Street Maps](https://www.openstreetmap.org/).

### Synthetic

Alternatively, synthetic data may be generated by running commands such as:

```
docker run gensit create ./data/inputs/configs/generic/synthetic_data_generation.toml \
-dim origin 100 -dim destination 30 -dim time 1 \
-sigma 0.0141421356 --synthesis_n_samples 1000 --synthesis_method sde_solver
```

The command above creates synthetic data based on the requirements in [the section above](#data) using `origin` and `destination` aritificial locations (in our case 100 origins, 30 destinations). A cost matrix is randomly generated for every OD pair. Destination attraction data is generated by running the `synthesis_method` for `synthesis_n_samples` steps (in our case by running the Harris Wilson SDE solver for 1000 steps).

You noticed that we load a configuration file named `synthetic_data_generation.toml` to achieve all this. We elaborate on the use of configs in the [next section](#configs).

## Configs

Configuration files contain all settings (key-value pairs) required to `run` NN-based or MCMC-based algorithms for learning the discrete origin-destination table and/or underlying continuous SIM parameters. They are stored in a `toml` format.

Each type of algorithm is associated with an `experiment` type . We hereby refer to the process of running one algorirthm for a given set of configuration parameters as `run`ning an `experiment`. Examples of experiments include `SIM_NN`, `SIM_MCMC`, `JointTableSIM_MCMC`, `DisjointTableSIM_NN`, and `JointTableSIM_NN`.

Most configuration keys can be `sweep`ed for each type of `experiment` being run. This means that a range of values over which the experiment will be run can be provided. For example, the `sigma` parameter below

```
[harris_wilson_model.parameters.sigma.sweep]
  default = 0.0141421356
  range = [0.0141421356, 0.1414213562, nan]
```

means that each experiment in the [`experiments`](#experiments) section will run with `sigma` = 0.0141421356 and `sigma` = 0.141421356. A `sweep` is therefore one run of an `experiment` over a unique set of config values. Sweeps can be either _isolated_ or _coupled_. The above example constitutes an isolated sweep. A coupled sweep is shown below:

```
[harris_wilson_model.parameters.sigma.sweep]
  default = 0.0141421356
  range = [0.0141421356, 0.1414213562, nan]
[training.to_learn.sweep]
  default = ['alpha', 'beta']
  range = [['alpha', 'beta'],['alpha', 'beta'],['alpha', 'beta', 'sigma']]
  coupled = true
  target_name = 'sigma'
```

Here `sigma` is coupled with the `to_learn` parameter, meaning the vary together. In this case each experiment will be run for three different sweep settings: (`sigma = 0.0141421356`, `to_learn` = ['alpha','beta']), (`sigma` = 0.1414213562, `to_learn` = ['alpha','beta']), and (`sigma` = nan, `to_learn` = ['alpha','beta','sigma']). We note that more than one sweep keys can be coupled.

> **_Note:_** More information on each key-value pair found in Configs can be found [here](./configuration_settings.md).

# Functionality

## Run

## Plot

## Summarise

## Reproduce

# Related publications

- Ioannis Zachos, Mark Girolami, Theodoros Damoulas. _Generating Origin-Destination Matrices in Neural Spatial Interaction Models_. (Under review).
- Ioannis Zachos, Theodoros Damoulas, Mark Girolami. _Table Inference for Combinatorial Origin-Destination Choices in Agent-based Population Synthesis_. [https://arxiv.org/abs/2307.02184](https://arxiv.org/abs/2307.02184) (Stat, 2024).

# Acknowledgments

We acknowledge support from the [UK Research and Innovation (UKRI) Research Council](https://www.ukri.org/), [Arup](https://www.arup.com/) and Cambridge University's [Center for Doctoral Training in Future Infrastructure and Built Environment](https://www.fibe-cdt.eng.cam.ac.uk/). We thank [Arup's City Modelling Lab](https://www.arup.com/services/digital/city-modelling-lab) for their insightful discussions and feedback without which this repository would not have come to fruition.
