# Neural DAEs: Constrained Neural Networks

This project contains the code used in https://arxiv.org/pdf/2211.14302.pdf.

The project introduces neural networks with constraints, which can be used to boost model inference significantly, while lowering constraint violations by several orders of magnitudes.
The project contains penalty constraints, auxiliary information regularization, and constraints through projections. 

The results for a 5-body pendulum are:

![CV_mean](https://github.com/tueboesen/Constrained-Neural-Networks/blob/main/figures/ntrain_10000_nskip_100_cv_mean.png)

![MAE_r](https://github.com/tueboesen/Constrained-Neural-Networks/blob/main/figures/ntrain_10000_nskip_100_mae_r.png)

## Installation

The requirements for running this code can be found in requirements.txt and can be installed by running:

```
pip install -r requirements.txt
```
Note that it might not always work to install the packages through the requirements.txt, since for instance Pytorch have special install requirements if you want gpu support.
In that case you should look at the individual packages website and see how they each should be installed.

## Project structure
The project uses MLflow for experiment tracking, Hydra for configuration file management and multiruns.
The core of this project is the constraint class, which can be found in src/constraints.py. 

## Getting started
As a starting point I would suggest running the multibody pendulum example "train_pendulum.py" in the examples folder.

### Reproducing paper results
The results found in the paper can easily be reproduced with train_pendulum.py and train_water.py.

train_paper is currently set to run pendulum predictions with 100 training samples and predictions 100 steps ahead for all constraint types, corresponding to the upper half of the first column of table 1.
In order to produce any of the other results in table 1, the appropriate settings are changed in configuration file multibodypendulum_tables.yaml.

Similarly, results in table two can be generated with train_water.py, though it should be noted that this takes significantly longer to run.

### Creating custom constraints

In order to create your own constraints create a class that inherits from the ConstraintTemplate class and define the required methods. 


## Contribution
The code in the project is running and has a few unit tests for the constraint class, which is the hearth of this project. Nevertheless, the project is still in a rather coarse and unpolished stage.
There are various TODOs spread throughout the project with things that could be relevant to implement that I simple haven't gotten around to. 
If anyone want to tackle any of the TODOs or have any specific requests for something they need, feel free to open an issue or code it up and send a pull request.
