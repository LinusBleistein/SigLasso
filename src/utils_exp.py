import json
import multiprocessing
import os
import pandas as pd
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


def extract_info(loc):
    """ Extracts the metrics from the directory. """
    infos = load_json(loc + '/info.json')
    # Strip of non-necessary entries
    # infos = {key: value['values'] for key, value in infos.items()}

    return infos


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

        df_list = []
        try:
            config = extract_config(loc)
            df_list.append(pd.DataFrame.from_dict(config, orient='index').T)

        except Exception as e:
            print('Could not load config at: {}. Failed with error:\n\t"{}"'.format(loc, e))
        try:
            metrics = extract_metrics(loc)
            df_list.append(pd.DataFrame.from_dict(metrics, orient='index').T)
        except Exception as e:
            print('Could not load metrics at: {}. Failed with error:\n\t"{}"'.format(loc, e))

        try:
            infos = extract_info(loc)
            df_list.append(pd.DataFrame.from_dict(infos, orient='index').T)
        except Exception as e:
            print('Could not load infos at: {}. Failed with error:\n\t"{}"'.format(loc, e))

        df = pd.concat(df_list, axis=1)
        df.index = [int(run_num)]
        frames.append(df)

    # Concat for a full frame
    df = pd.concat(frames, axis=0, sort=True)
    df.sort_index(inplace=True)

    return df
