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
- [Problem setup](#problem-setup)
- [Functionality](#functionality)
  - [Run](#run)
  - [Plot](#plot)
  - [Summarise](#summarise)
  - [Reproduce](#reproduce)
- [Conclusion](#conclusion)
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

# Problem setup

Consider $M$ agents that travel from $I$ origins to $J$ destinations to work. Let the expected number of trips (intensity) of agents between origin $i$ and destination $j$ be denoted by $\Lambda_{ij}$. The residential population in each origin (row sums) is equal to

$$
    \Lambda_{i+} = \sum_{j=1}^{J} \Lambda_{ij}, \;\;\;\; i=1,\dots,I,
$$

while the working population at each destination (column sums) is

$$
    \Lambda_{+j} = \sum_{i=1}^{I} \Lambda_{ij}, \;\;\;\; j=1,\dots,J.
$$

We assume that the total origin and destination demand are both conserved:

$$
    M = \Lambda_{++} = \sum_{i=1}^{I} \Lambda_{i+} = \sum_{j=1}^{J} \Lambda_{+j}.
$$

The demand for destination zones depends on the destination's attractiveness denoted by $\mathbf{z} = (z_1,\dots, z_J) \in \mathbb{R}_{>0}^{J}$. Let the log-attraction be $\mathbf{x} = \log(\mathbf{z})$. Between two destinations of similar attractiveness, agents are assumed to prefer nearby zones. Therefore, a cost matrix $\mathbf{C} = (c_{i,j})_{i,j=1}^{I,J}$ is introduced to reflect travel impedance. The maximum entropy distribution of agent trips subject to the total number of agents being conserved is derived by maximising the Lagrangian

$$
\mathcal{E}(\boldsymbol{\Lambda}) = \sum_{i=1}^{I}\sum_{j=1}^J -\Lambda_{ij}\log(\Lambda_{ij}) - \zeta \sum_{i,j}^{I,J} \Lambda_{ij} + \alpha \sum_{j}^{J} x_j\Lambda_{ij}  - \beta \sum_{i,j}^{I,J} c_{ij}\Lambda_{ij},
$$

where $\zeta,\alpha$ and $\beta$ are the Lagrange multipliers. This yields a closed-form expression for the expected flows (intensity) of agents from $i$ to $j$ for the total constrained SIM:

$$
\Lambda_{ij} = \frac{\Lambda_{++}\exp(\alpha x_j -\beta c_{ij})}{\sum_{k,m}^{I,J} \exp(\alpha x_m-\beta c_{km})},
$$

where the multipliers $\alpha,\beta$ control the two competing forces of attractiveness and deterrence, respectively, while $\zeta$ bears no physical interpretation. A higher $\alpha$ relative to $\beta$ characterises a preference over destinations with higher job availability, while the contrary indicates a predilection for closer workplaces. If we further assume that origin demand ($\Lambda_{.+}$) is also fixed then we get the singly constrained SIM:

$$
\Lambda_{ij} = \frac{\Lambda_{i+}\exp(\alpha x_j -\beta c_{ij})}{\sum_{k,m}^{I,J} \exp(\alpha x_m-\beta c_{km})}.
$$

Spatial interaction models are connected to physics models through the destination attractiveness term $\mathbf{z}$, which is governed by the Harris-Wilson system of $J$ coupled ordinary differential equations (ODEs):

where $\epsilon$ is a responsiveness parameter, $\kappa>0$ is the number of agents competing for one job, $\delta>0$ is the smallest number of jobs a destination can have and $\Lambda_{+j}(t) - \kappa z_j(t)$ is the net job capacity in destination $j$. A positive net job capacity translates to a higher economic activity (more travellers than jobs) and a boost in local employment, and vice versa. A stochastic version of the Harris Wilson model is the following:

$$
\frac{dz_j}{dt} = \epsilon z_j \left( \Lambda_{+j} - \kappa z_j + \delta  \right) + \sigma z_j \circ B_{j,t}, \; \mathbf{z}(0) = \mathbf{z}'.
$$

We recommend you look at relevant [publications](#related-publications) for more information on the Harris Wilson model. Our first goal is to learn the parameters $\alpha,\beta$ using either sampling (MCMC) or optimisation (NN) algorithms. To achieve this goal we leverage data $\mathcal{D}$ about either the observed destination attraction (e.g. the number of jobs available at each destination) and/or the total distance/time traveled by agents in their work trips. In general, we can say that our first goal is to learn the distribution of the agent trip intensity $p(\boldsymbol{\Lambda}\vert \mathcal{C},\mathcal{D})$.

We note that the discrete number of agents traveling to work is represented by

$$
T_{ij} \sim \text{Poisson}(\Lambda_{ij}).
$$

Although $\mathbf{T}$ and $\boldsymbol{\Lambda}$ look like similar quantities we emphasize that they are distinct. The former is a discrete quantity while the latter is a continuous quantity and many $T_{ij}$ may be ''plausible'' under a single $\Lambda_{ij}$. The SIM intensity $\boldsymbol{\Lambda}$ is a mean-field approximation and can be thought of as the expectation (average) of $\mathbf{T}$ across time for all work trips. We can also reason at a probability level by thinking of $0 \leq \Lambda_{ij}/\Lambda_{++}\leq 1$ as transition probabilities from an origin $i$ to a destination $j$. Depending on the available summary data (e.g. $\mathbf{T}_{.+},\mathbf{T}_{+.}$) we define a set of constraints $\mathcal{C}$. Our second goal is to sample $\mathbf{T}$ subject to these constraints, i.e. sample from $p(\mathbf{T}\vert \mathcal{C},\mathcal{D})$. More information is provided in the papers provided in the [publications](#related-publications) section.

# Functionality

The `GeNSIT` package provides functionality for five different operations: [`create`](#synthetic), [`run`](#run), [`plot`](#plot), [`reproduce`](#reproduce), [`summarise`](#summarise).

## Run

This command runs `experiment`s using Markov Chain Monte Carlo and/or Neural Networks based on a `Config` file. For example, we can run joint table and intensity inference using the following command

```
docker run gensit run ./data/inputs/configs/generic/joint_table_sim_inference.toml \
-et JointTableSIM_NN -nw 6 -nt 3
```

This config runs a `JointTableSIM_NN` experiment using 6 number of workers and 3 number of threads per worker. A list of experiments and the types of algorithms they use to learn $\boldsymbol{\Lambda}$, $\mathbf{T}$ is provided below. We note that experiments that use NNs to learn the intensity function parameters tend to be computationally much faster.

| Experiment            | $\mathbf{T}$ | $\boldsymbol{\Lambda}$ |
| --------------------- | :----------: | :--------------------: |
| `SIM_MCMC`            |      -       |          MCMC          |
| `JointTableSIM_MCMC`  |     MCMC     |          MCMC          |
| `SIM_NN`              |      -       |           NN           |
| `DisjointTableSIM_NN` |     MCMC     |           NN           |
| `JointTableSIM_NN`    |     MCMC     |           NN           |

The `run` command can also be programmatically executed using the notebook [Example 1 - Running experiments](../notebooks/Example%201%20-%20Running%20experiments.ipynb).

## Plot

Once an experiment has been completed, we can use the following command to plot its data:

```
docker run gensit plot [PLOT_VIEW] [PLOT_TYPE] -x [X_DATA] -y [Y_DATA]
```

where `PLOT_VIEW` defines the type of view the data should be shown. Views can be simple, tabular or spatial. `PLOT_TYPE` can be either line or scatter. The `X_DATA` or `Y_DATA` are provided as names of experiment outputs or their evaluated expressions (see [Config settings](./configuration_settings.md)).

For example, the code below plots the log destination attraction predictions (x-axis) against the observed data (y-axis) for experiments `JointTableSIM_MCMC`,`JointTableSIM_NN`,`NonJointTableSIM_NN`.

```
docker run gensit plot simple scatter \
-y log_destination_attraction_data -x mean_log_destination_attraction_predictions \
-dn cambridge_work_commuter_lsoas_to_msoas/exp1 \
-et JointTableSIM_MCMC -et JointTableSIM_NN -et NonJointTableSIM_NN \
-el np -el xr -el MathUtils \
-e mean_log_destination_attraction_predictions "signed_mean_func(log_destination_attraction,sign,dim=['id']).squeeze('time')" \
-e mean_log_destination_attraction_predictions "log_destination_attraction.mean('id').squeeze('time')" \
-e log_destination_attraction_data "np.log(destination_attraction_ts).squeeze('time')" \
-ea log_destination_attraction -ea sign \
-ea "destination_attraction_ts=outputs.inputs.data.destination_attraction_ts" \
-ea "signed_mean_func=MathUtils.signed_mean" \
-k sigma -k title \
-cs "da.loss_name.isin([str(['dest_attraction_ts_likelihood_loss']),str(['dest_attraction_ts_likelihood_loss', 'table_likelihood_loss'])])" \
-cs "~da.title.isin(['_unconstrained','_total_constrained','_total_intensity_row_table_constrained'])" \
-c title -op 1.0 -mrkr sigma -l title -l sigma -msz 20 \
-ft 'predictions_figure/destination_attraction_predictions_vs_observations' \
-xlab '$\mathbb{E}\left[\mathbf{x}^{(1:N)}\right]$' \
-ylab '$\mathbf{y}$'
```

The `-e`,`-ea`,`-el` arguments define the evaluated expressions, the keyword arguments used as input to these expressions (also evaluated) and the necessary libraries that are used to perform the operations, respectively. The evaluation is performed using Python's `eval` function. The first two types of argument allow for reading input and/or output data directly. For example, `-ea "destination_attraction_ts=outputs.inputs.data.destination_attraction_ts"` loads the input (observed) destination attraction time series data while `-ea log_destination_attraction -ea sign` loads `log_destination_attraction` and `sign` output datasets.

The output data is sliced using the coordinate values specified by the `-cs` arguments. For instance, `-cs "~da.title.isin(['_unconstrained','_total_constrained','_total_intensity_row_table_constrained'])"` only keeps the datasets whose `title` variable is equal to any of the specified values. The `sweep` data are gathered either from the output dataset itself or from the output config file (in this case we elicit `sigma`,`title` `sweep` variables).

The scatter plot is colored by the `title` variable and its markers are determined by the `sigma` variable. Both of these variables are contained in each `sweep` that was run. The exact mappings from say sigma values to marker types are contained in [this file](../gensit/static/plot_variables.py). Each point is labeled by both the `title` and `sigma` values. The resulting figure is shown below.

<img src="./example_figure.jpg" alt="framework" width="500"/>

## Summarise

This command summarised the output data and creates a `csv` file with each data summary from every `sweep`. For example, if we wish to compute the Standardised Root Mean Square Error (SRMSE) for `JointTableSIM_NN` we run

```
docker run gensit summarise \
-dn cambridge_work_commuter_lsoas_to_msoas/exp1 \
-et JointTableSIM_NN \
-el np -el MathUtils -el xr \
-e table_srmse "srmse_func(prediction=mean_table,ground_truth=ground_truth)" \
-e intensity_srmse "srmse_func(prediction=mean_intensity,ground_truth=ground_truth)" \
-ea table -ea intensity -ea sign \
-ea "srmse_func=MathUtils.srmse" \
-ea "signed_mean_func=MathUtils.signed_mean" \
-ea "ground_truth=outputs.inputs.data.ground_truth_table" \
-ea "mean_table=table.mean(['id'])" \
-ea "mean_intensity=signed_mean_func(intensity,'intensity','signedmean',dim=['id'])" \
-ea "mean_intensity=intensity.mean(['id'])" \
-cs "da.loss_name.isin([str(['dest_attraction_ts_likelihood_loss']),str(['dest_attraction_ts_likelihood_loss', 'table_likelihood_loss']),str(['table_likelihood_loss'])])" \
-btt 'iter' 10000 90 1000 \
-k sigma -k type -k name -k title -fe SRMSEs -nw 20
```

The arguments are similar to the `plot` command. Here we also use `-btt` refered to as burning, thinning and trimming to slice the `iter` coordinate values based on their index. In this occasion, we discard the first 10000 samples and then only keep every 90th sample. Finally, we trim this data array to 1000 elements. A small part of the summarised table is shown below.

| type             | sigma                | title                                   | name               | proposal        | intensity_srmse      | table_srmse           |
| ---------------- | -------------------- | --------------------------------------- | ------------------ | --------------- | -------------------- | --------------------- |
| JointTableSIM_NN | 0.141                | \_doubly_20%\_cell_constrained          | TotallyConstrained | degree_higher   | [1.983129620552063]  | [0.37612101435661316] |
| JointTableSIM_NN | 0.014139999635517597 | \_unconstrained                         | TotallyConstrained | direct_sampling | [29.513639450073242] | [1.7274982929229736]  |
| JointTableSIM_NN | 0.14142000675201416  | \_unconstrained                         | TotallyConstrained | direct_sampling | [29.513639450073242] | [1.7274982929229736]  |
| JointTableSIM_NN | 0.14142000675201416  | \_doubly_constrained                    | TotallyConstrained | degree_higher   | [2.019375801086426]  | [0.4618358910083771]  |
| JointTableSIM_NN | 0.014139999635517597 | \_doubly_10%\_cell_constrained          | TotallyConstrained | degree_higher   | [0.8903818130493164] | [0.42363834381103516] |
| JointTableSIM_NN | 0.14142000675201416  | \_total_intensity_row_table_constrained | TotallyConstrained | direct_sampling | [5.928720951080322]  | [2.1622815132141113]  |
| JointTableSIM_NN | 0.014139999635517597 | \_doubly_20%\_cell_constrained          | TotallyConstrained | degree_higher   | [0.9381942749023438] | [0.37765341997146606] |
| JointTableSIM_NN | 0.14142000675201416  | \_doubly_constrained                    | TotallyConstrained | degree_higher   | [0.6841356754302979] | [0.5531010627746582]  |
| JointTableSIM_NN | 0.14142000675201416  | \_unconstrained                         | TotallyConstrained | direct_sampling | [29.513639450073242] | [1.7273409366607666]  |

Processing experimental outputs for uses similar to the ones provided by `plot` and `summarise` commands can also be achieved by following the steps of notebook [Example 2 - Reading outputs](../notebooks/Example%202%20-%20Reading%20outputs.ipynb).

## Reproduce

Finally, this command is run to reproduce the figures appearing in the [papers](#related-publications). The commands are self-explanatory:

```
docker run gensit reproduce figure1;
docker run gensit reproduce figure2;
docker run gensit reproduce figure3;
docker run gensit reproduce figure4;
```

# Conclusion

We have introduced `GeNSIT`, an efficient framework for sampling jointly the discrete combinatorial space of agent trips ($\mathbf{T}$) subject to summary statistic data $\mathcal{C}$ and its mean-field $\boldsymbol{\Lambda}$ limit. Therefore, users of this package can perform agent location choice synthesis based on the available data $\mathcal{C},\mathcal{D}$. Although our discussion has been limited to residence to work trips, other types of trips could be modelled too, such as residence to shopping center. The main limitations of this package are the inability to model **activity chains** as opposed to trips and the fact that only static (time-independent) origin destination matrices are considered.

# Related publications

- Ioannis Zachos, Mark Girolami, Theodoros Damoulas. _Generating Origin-Destination Matrices in Neural Spatial Interaction Models_. (Under review).
- Ioannis Zachos, Theodoros Damoulas, Mark Girolami. _Table Inference for Combinatorial Origin-Destination Choices in Agent-based Population Synthesis_. [https://arxiv.org/abs/2307.02184](https://arxiv.org/abs/2307.02184) (Stat, 2024).

# Acknowledgments

We acknowledge support from [Arup](https://www.arup.com/), the [UK Research and Innovation (UKRI) Research Council](https://www.ukri.org/) and Cambridge University's [Center for Doctoral Training in Future Infrastructure and Built Environment](https://www.fibe-cdt.eng.cam.ac.uk/). We thank [Arup's City Modelling Lab](https://www.arup.com/services/digital/city-modelling-lab) for their insightful discussions and feedback without which this project would not have come to fruition.

Thank you for visiting our GitHub repository! We're thrilled to have you here. If you find our project useful or interesting, please consider showing your support by starring the repository and forking it to explore its features and contribute to its development. Your support means a lot to us and helps us grow the community around this project. If you have any questions or feedback, feel free to open an issue or reach out to us.
