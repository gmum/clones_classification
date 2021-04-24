import yaml
from torch import nn

from data.bacteria_dataset import BacteriaBag


def load_bacteria_cv(labels, patches, features, cv, curr_class, shuffle):
    train_dataset = BacteriaBag(labels,
                                patches,
                                features,
                                cv, curr_class, 'train', shuffle)
    valid_dataset = BacteriaBag(labels,
                                patches,
                                features,
                                cv, curr_class, 'test', shuffle)
    test_dataset = BacteriaBag(labels,
                                patches,
                                features,
                                cv, curr_class, 'test', shuffle)
    return train_dataset, valid_dataset, test_dataset


def read_config(config_path: str) -> dict:
    with open(config_path, 'r') as config_file:
        config = yaml.load(config_file)
    return config


def get_loss(loss_function):
    if loss_function == 'MSELoss':
        return nn.MSELoss()
    elif loss_function == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss()
    elif loss_function == 'BCELoss':
        return nn.BCELoss()
    elif loss_function == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()
    else:
        print('No such loss function')