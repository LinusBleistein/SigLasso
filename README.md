# Learning the Dynamics of Sparsely Observed Interacting Systems ([arXiv](https://arxiv.org/pdf/2301.11647.pdf))

## Requirements

For reproductibility purposes, we advise to install the project in a dedicated virtual environment to 
make sure the specific requirements are satisfied. Recommended Python version: 3.7.x.

To install requirements, use

``pip install -r requirements.txt``

Then install our code as a package using

``pip install -e .`` 

## Running an experiment

This code is based on the [sacred](https://sacred.readthedocs.io/en/stable/quickstart.html) framework.  To run an experiment, you need to 
specify a configuration in `experiments/configs.py` and then run 

``python experiments/simulation_main.py name_config``

For example, to reproduce the experiment of the paper where we vary the number 
of sampling points of the target time series (Figure 2), run

``python experiments/simulation_main.py y_sampling``

The results of the experiments are saved in a folder. The notebook 
`notebooks/Analyse results.ipynb` illustrates how to then load the results 
from the experiment files.

See also the notebook `notebooks/Demo models.ipynb` for some simple examples.

## Reproducing the paper experiments

All hyperparameters used in the article are given in the configuration file 
`experiments/configs.py`. 

## Toy experiment



## HGR forecasting during the COVID-19 pandemic

Please refer to the notebook `notebooks/Covid preprocessing.ipynb` for details on the pre-processing of the COVID-19 data and to `notebooks/Covid results plots.ipynb` for result analysis. The original data can be found [here](https://gitlab.pasteur.fr/mmmi-pasteur/covid19-ensemble-model).  
