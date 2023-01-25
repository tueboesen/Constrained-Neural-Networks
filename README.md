# Constrained Neural Networks

This project tests various ways of adding constraints into a neural network, as detailed in https://arxiv.org/pdf/2211.14302.pdf.

## Installation

The requirements for running this code can be found in requirements.txt and can be installed by running:

```
pip install -r requirements.txt
```

Note that it might not always work to install the packages through the requirements.txt, since for instance Pytorch have special install requirements if you want gpu support.
In that case you should look at the individual packages website and see how they each should be installed.

## Project structure
### Data
Data goes here. For instance water.npz which is not included in the github project due to size limitations. But the water.npz dataset for water simulations can be requested by sending an email to the corresponding author of tha paper.

### Examples
Scripts to run. 
The subfolder Paper_results contains scripts used in the generation of the paper results.

### postprocess
Scripts used to evaluate the results and produce tables or figures for the paper. 

### preprocess
Scripts that were used in order to get data into a format used in this project. Currently contains scripts for converting a cp2k dataset to a npz file, and for preparing proteinnet datasets.

### scripts_to_run
Empty folder. If you want to run many scripts sequentially, you can put them in here and run the run_all_scripts.py file from the examples folder. 

### src
The source code folder.

### verlet_integration
scripts for running a Verlet integration using a pretrained neural network for predicting future MD states.

## Running it
As a starting point I would suggest running some of the scripts in Examples

The code can be run from the commandline if desired, but generally I would suggest that you start by running the code through one of the pre-made examples, and then modify one of those to fit your need.

Pendulum simulations can be run without a dataset, since the source code includes a multi-body pendulum simulator that will generate datasets on the fly.


