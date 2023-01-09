import json
import os
import pandas as pd
from sacred.observers import FileStorageObserver
from sklearn.model_selection import ParameterGrid
import torch

from src.datagen import get_train_val_test


def simulate_and_save_data(data_config, data_path):
    X_raw_train, Y_raw_train, X_raw_val, Y_raw_val, X_raw_test, Y_raw_test = \
        get_train_val_test(
            data_config['model_X'],
            data_config['model_Y'],
            data_config['n_train'],
            data_config['n_val'],
            data_config['n_test'],
            data_config['n_points_true'],
            dim_X=data_config['dim_X'],
            dim_Y=data_config['dim_Y'],
            non_linearity_Y=data_config['non_linearity_Y']
        )

    torch.save(X_raw_train, f"{data_path}/X_raw_train.pt")
    torch.save(Y_raw_train, f"{data_path}/Y_raw_train.pt")
    torch.save(X_raw_val, f"{data_path}/X_raw_val.pt")
    torch.save(Y_raw_val, f"{data_path}/Y_raw_val.pt")
    torch.save(X_raw_test, f"{data_path}/X_raw_test.pt")
    torch.save(Y_raw_test, f"{data_path}/Y_raw_test.pt")
    
    return data_path


def load_data(data_path):
    X_raw_train = torch.load(f"{data_path}/X_raw_train.pt")
    Y_raw_train = torch.load(f"{data_path}/Y_raw_train.pt")
    X_raw_val = torch.load(f"{data_path}/X_raw_val.pt")
    Y_raw_val = torch.load(f"{data_path}/Y_raw_val.pt")
    X_raw_test = torch.load(f"{data_path}/X_raw_test.pt")
    Y_raw_test = torch.load(f"{data_path}/Y_raw_test.pt")
    
    return (X_raw_train, Y_raw_train, X_raw_val, Y_raw_val, X_raw_test, 
            Y_raw_test)

#TODO: move to experiment_1 and rename: this is the main function and should not be in utils

def gridsearch(ex, config, BASE_DIR):
    """Loops over all the experiments in a configuration grid.
    Parameters
    ----------
        ex: object
            Instance of sacred.Experiment()
        config_grid: dict
            Dictionary of parameters of the experiment.
        niter: int, default=10
            Number of iterations of each experiment
        dirname: str, default='my_runs'
            Location of the directory where the experiments outputs are stored.
    """
    niter = config['niter']

    results_path = os.path.join(BASE_DIR, f"results/{config['name']}")
    data_path = os.path.join(BASE_DIR, f"data/{config['name']}")

    os.makedirs(results_path, exist_ok=True)
    os.makedirs(data_path, exist_ok=True)

    ex.observers.append(FileStorageObserver(results_path))
    exp_grid = list(ParameterGrid(config['exp_config']))

    for i in range(niter):
        simulate_and_save_data(config['data_config'], data_path)
        for params in exp_grid:
            params['data_path'] = data_path
            ex.run(config_updates=params, info={})


def load_json(path):
    """Loads a json object
    Parameters
    ----------
    path: str
        Location of the json file.
    """
    with open(path) as file:
        return json.load(file)


def extract_config(loc):
    """ Extracts the metrics from the directory."""
    config = load_json(loc + '/config.json')
    return config


def extract_metrics(loc):
    """ Extracts the metrics from the directory. """
    metrics = load_json(loc + '/metrics.json')

    # Strip of non-necessary entries
    metrics = {key: value['values'] for key, value in metrics.items()}

    return metrics


def get_ex_results(dirname):
    """Extract all result of a configuration grid.
    Parameters
    ----------
    dirname: str
        Name of the directory where the experiments are stored.
    Returns
    -------
    df: pandas DataFrame
        Dataframe with all the experiments results
    """
    not_in = ['_sources', '.DS_Store']
    # dir_path = os.path.dirname(os.path.realpath(__file__))
    # dirname = dir_path + '/' + dirname
    run_nums = [x for x in os.listdir(dirname) if x not in not_in]

    frames = []
    for run_num in run_nums:
        loc = dirname + '/' + run_num
        try:
            config = extract_config(loc)
        except Exception as e:
            print('Could not load config at: {}. Failed with error:\n\t"{}"'.format(loc, e))
        try:
            metrics = extract_metrics(loc)
        except Exception as e:
            print('Could not load metrics at: {}. Failed with error:\n\t"{}"'.format(loc, e))

        # Create a config and metrics frame and concat them
        config = {str(k): str(v) for k, v in config.items()}    # Some dicts break for some reason
        df_config = pd.DataFrame.from_dict(config, orient='index').T
        df_metrics = pd.DataFrame.from_dict(metrics, orient='index').T

        df = pd.concat([df_config, df_metrics], axis=1)
        df.index = [int(run_num)]
        frames.append(df)

    # Concat for a full frame
    df = pd.concat(frames, axis=0, sort=True)
    df.sort_index(inplace=True)

    return df
