import datetime
import os
import sys

from munch import munchify
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from utils.bacteria_utils import load_bacteria_cv, read_config
from model.experiments import experiment
from model.CNN import CNN as Model


kwargs = {'num_workers': 0, 'pin_memory': True}


def run(config, kwargs):
    config['model_signature'] = str(datetime.datetime.now())[0:19]

    model_name = '' + config['model_name']

    if config['loc_gauss'] or config['loc_inv_q'] or config['loc_att']:
        config['loc_info'] = True

    if config['att_gauss_abnormal'] or config['att_inv_q_abnormal'] or config['att_gauss_spatial'] or \
            config['att_inv_q_spatial'] or config['att_module']:
        config['self_att'] = True

    print(config)

    with open('experiment_log_' + config['operator'] + '.txt', 'a') as f:
        print(config, file=f)

    print('\nSTART KFOLDS CROSS VALIDATION\n')

    train_error_folds = []
    test_error_folds = []
    labels = pd.read_csv(config['labels_filename'], index_col=0)
    patches = pd.read_csv(config['patches_filename'], index_col=0)
    features = torch.load(config['features_filename'])
    curr_class = config['curr_class']
    curr_fold = config['curr_fold']
    for current_fold in [curr_fold]:

        print('################ Train-Test fold: {}/{} #################'.format(current_fold + 1, config['kfold']))

        snapshots_path = 'snapshots/'
        dir = snapshots_path + model_name + '_' + config['model_signature'] + '/'
        sw = SummaryWriter(f"tensorboard/{model_name}_{config['model_signature']}_fold_{current_fold}")

        if not os.path.exists(dir):
            os.makedirs(dir)

        train_set, val_set, test_set = load_bacteria_cv(labels,
                                                        patches,
                                                        features,
                                                        config['split'],
                                                        curr_class,
                                                        shuffle=True)
        clss, counts = np.unique(train_set.label_list, return_counts=True)
        counts = 1 - counts / np.sum(counts)
        class_counts = {int(clss[c]): counts[c] for c in range(len(clss))}
        train_sampleweights = [class_counts[int(y_bi)] for y_bi in train_set.label_list]
        sampler = WeightedRandomSampler(
            weights=train_sampleweights,
            num_samples=len(train_sampleweights),
        )

        print('\tcreate models')
        args = munchify(config)
        args.activation = nn.ReLU()
        model = Model(args)
        model.cuda(config['device'])

        print('\tinit optimizer')
        if config['optimizer'] == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.999), weight_decay=config['reg'])
        elif config['optimizer'] == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=config['lr'], weight_decay=config['reg'], momentum=0.9)
        else:
            raise Exception('Wrong name of the optimizer!')

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=45, gamma=0.1)

        print('\tperform experiment\n')

        train_error, test_error = experiment(
            args,
            kwargs,
            current_fold,
            train_set,
            val_set,
            test_set,
            sampler,
            model,
            optimizer,
            scheduler,
            dir,
            sw,
        )

        train_error_folds.append(train_error)
        test_error_folds.append(test_error)

        with open('final_results_' + config['operator'] + '.txt', 'a') as f:
            print('Class: {}\n'
                  'RESULT FOR A SINGLE FOLD\n'
                  'SEED: {}\n'
                  'OPERATOR: {}\n'
                  'FOLD: {}\n'
                  'ERROR (TRAIN): {}\n'
                  'ERROR (TEST): {}\n\n'.format(curr_class,
                                                config['seed'],
                                                config['operator'],
                                                current_fold,
                                                train_error,
                                                test_error),
                  file=f)
    # ==================================================================================================================
    print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
    with open('experiment_log_' + config['operator'] + '.txt', 'a') as f:
        print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n', file=f)

    return np.mean(train_error_folds), np.std(train_error_folds), np.mean(test_error_folds), np.std(test_error_folds)


if __name__ == "__main__":
    config_file = sys.argv[1]
    config = read_config(config_file)
    seed = config['seed']

    train_mean_list = []
    test_mean_list = []

    torch.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    lr = config['lr']
    reg = config['reg']
    train_mean, train_std, test_mean, test_std = run(config, kwargs)

    with open('final_results_' + config['model_name'] + '.txt', 'a') as f:
        print('lr = {}, reg = {}\n'
              'RESULT FOR A SINGLE SEED, 5 FOLDS\n'
              'SEED: {}\n'
              'OPERATOR: {}\n'
              'TRAIN MEAN {} AND STD {}\n'
              'TEST MEAN {} AND STD {}\n\n'.format(lr,
                                                   reg,
                                                   seed,
                                                   config['operator'],
                                                   train_mean,
                                                   train_std,
                                                   test_mean,
                                                   test_std),
              file=f)

    train_mean_list.append(train_mean)
    test_mean_list.append(test_mean)

    with open('final_results_' + config['operator'] + '.txt', 'a') as f:
        print('RESULT FOR 1 SEEDS, 5 FOLDS\n'
              'OPERATOR: {}\n'
              'TRAIN MEAN {} AND STD {}\n'
              'TEST MEAN {} AND STD {}\n\n'.format(config['operator'],
                                                   np.mean(train_mean_list),
                                                   np.std(train_mean_list),
                                                   np.mean(test_mean_list),
                                                   np.std(test_mean_list)),
              file=f)

    with open('final_results_' + config['operator'] + '.txt', 'a') as f:
        print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n', file=f)
