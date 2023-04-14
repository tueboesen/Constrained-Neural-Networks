# Constrained Neural Networks

This project tests various ways of adding constraints into a neural network, as detailed in https://arxiv.org/pdf/2211.14302.pdf.

We introduce several constraint methods and compare those with previous methods as well as no constraints.

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
The project uses MLflow for experiment tracking, Hydra for configuration file management and multiruns and Optuna for optimization.
The core of this project is the constraint class, which can be found in src/constraints.py. In order to create your own constraints you inherit from the ConstraintTemplate class and define the required methods.  

## Getting started
As a starting point I would suggest running the multibody pendulum example "train_pendulum.py" in the examples folder.




