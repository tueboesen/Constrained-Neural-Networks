# Neural DAEs: Constrained Neural Networks

This project contains the code used to generate the results in the paper: https://arxiv.org/pdf/2211.14302.pdf. A work authored by Tue Boesen, Eldad Haber, and Uri M. Ascher.

The project introduces neural networks with constraints, which can be used to boost model inference significantly, while lowering constraint violations by several orders of magnitudes.
The project contains penalty constraints, auxiliary information regularization, and constraints through projections. 

Adding constraints to a neural network trained on a 5-body pendulum generally gives significant improvements in prediction accuracy, and lowers constraint violations by several orders of magnitudes:

![CV_mean](https://github.com/tueboesen/Constrained-Neural-Networks/blob/main/figures/ntrain_10000_nskip_100_cv_mean.png)

![MAE_r](https://github.com/tueboesen/Constrained-Neural-Networks/blob/main/figures/ntrain_10000_nskip_100_mae_r.png)

(more details can be found in the paper linked above)

## Installation

The requirements for the project can be found in pyproject.toml, and is known to work with poetry.

Note that the package torch_cluster should be downloaded from source and might cause trouble during an automatic poetry install. If this is the case, check the torch_cluster website for additional information on how to install it in the future.
Furthermore, it should be noted that the code is only tested on Ubuntu. 


## Project structure
The project uses MLflow for experiment tracking, Hydra for configuration file management and multiruns.
The core of this project is the constraint class, which can be found in src/constraints.py. 

## Getting started
As a starting point I would suggest running the multibody pendulum example `./examples/train_pendulum.py`.

## Reproducing paper results
The code has a tagged release called paper version. This is the code that was used to produce the paper results. Versions after that might not produce the results exactly, but should generally produce similar results. 
The results found in the paper can easily be reproduced with `./examples/train_pendulum.py`, `./examples/train_water.py`, and `./examples/train_imagedenoising.py`. 

### Multibody pendulum
`./examples/train_pendulum.py` is currently set to run pendulum predictions with 100 training samples and predictions 100 steps ahead for all constraint types, corresponding to the upper half of the first column of table 1.
In order to produce any of the other results in table 1, the below settings are changed in configuration file `./config/multibodypendulum_tables.yaml` to the appropriate values given in the table.

```
  n_train: 
  n_val: 
  nskip: 
```

### Water molecules
The results in the water molecule experiment can be generated using t`./examples/train_water.py`, though it should be noted that this takes significantly longer to run. 
The dataset required for `train_water.py` is too large for Github, but can be found using [DOI:10.5281/zenodo.7981759](https://zenodo.org/doi/10.5281/zenodo.7981759), and needs to be downloaded and unpacked to ./data . A script for downloading and unpacking the water dataset is found in `./data/download_water_data.py`
In order to produce any of the other results in table 2, the appropriate settings are changed in configuration file `./config/water_tables.yaml`.

### Vector field denoising
`./examples/train_imagedenoising.py` runs the vector field denoising example. 
The vector field data used in the paper is already in the repository in `/data/imagedenoising/images.npz`, but otherwise new data can also be generated using `/src/imagedenoising/create_imagedenoising_data.py`.

### Creating custom constraints

In order to create your own constraints create a class that inherits from the ConstraintTemplate class and define the required methods. 


## Contribution
The code in the project is running and has a few unit tests for the constraint class, which is the hearth of this project. Nevertheless, the project is still in a rather coarse and unpolished stage.
There are various TODOs spread throughout the project with things that could be relevant to implement that I simple haven't gotten around to. 
If anyone want to tackle any of the TODOs or have any specific requests for something they need, feel free to open an issue or code it up and send a pull request.
