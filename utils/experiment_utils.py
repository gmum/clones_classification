import numpy as np
import torch
from torch.autograd import Variable

import time


def train(args, train_loader, model, optimizer):
    train_loss = 0.
    train_error = 0.

    model.run_train(True)

    for batch_idx, (data, label) in enumerate(train_loader):
        label = label[0]
        if args.cuda:
            data, label = data.cuda(), label.cuda()
        data, label = Variable(data), Variable(label)

        optimizer.zero_grad()
        loss, gamma, gamma_kernel = model.calculate_objective(data, label)
        train_loss += loss[0]
        train_error += model.calculate_classification_error(data, label)[0]
        loss.backward()
        optimizer.step()

    train_loss /= len(train_loader)
    train_error /= len(train_loader)

    return model, train_loss, train_error, gamma, gamma_kernel


def evaluate(args, model, train_loader, data_loader, mode):
    model.eval()

    if mode == 'validation':
        evaluate_loss = 0.
        evaluate_error = 0.
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(data_loader):
                if len(label) > 1:
                    for i in range(len(label)):
                        data_one = data[i]
                        label_one = label[i]
                        if args.cuda:
                            data_one, label_one = data_one.cuda(), label_one.cuda()
                        data_one, label_one = Variable(data_one), Variable(label_one)
                        evaluate_loss += model.calculate_objective(data_one, label_one)[0]/len(label)
                        evaluate_error += model.calculate_classification_error(data_one, label_one)[0]/len(label)
                else:
                    data_one = data[0]
                    label_one = label[0]
                    if args.cuda:
                        data_one, label_one = data_one.cuda(), label_one.cuda()
                    data_one, label_one = Variable(data_one), Variable(label_one)
                    evaluate_loss += model.calculate_objective(data_one, label_one)[0] / len(label)
                    evaluate_error += model.calculate_classification_error(data_one, label_one)[0] / len(label)

        evaluate_loss /= len(data_loader)
        evaluate_error /= len(data_loader)

    if mode == 'test':
        train_error = 0.
        train_loss = 0.
        evaluate_error = 0.
        evaluate_loss = 0.
        t_ll_s = time.time()
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(data_loader):
                if len(label) > 1:
                    for i in range(len(label)):
                        data_one = data[i]
                        label_one = label[i]
                        if args.cuda:
                            data_one, label_one = data_one.cuda(), label_one.cuda()
                        data_one, label_one = Variable(data_one), Variable(label_one)
                        evaluate_loss += model.calculate_objective(data_one, label_one)[0] / len(label)
                        evaluate_error += model.calculate_classification_error(data_one, label_one)[0] / len(label)
                else:
                    data_one = data[0]
                    label_one = label[0]
                    if args.cuda:
                        data_one, label_one = data_one.cuda(), label_one.cuda()
                    data_one, label_one = Variable(data_one), Variable(label_one)
                    evaluate_loss += model.calculate_objective(data_one, label_one)[0] / len(label)
                    evaluate_error += model.calculate_classification_error(data_one, label_one)[0] / len(label)

        t_ll_e = time.time()
        evaluate_error /= len(data_loader)
        evaluate_loss /= len(data_loader)
        print('\tTEST classification error value (time): {:.4f} ({:.2f}s)'.format(evaluate_error, t_ll_e - t_ll_s))
        print('\tTEST log-likelihood value (time): {:.4f} ({:.2f}s)\n'.format(evaluate_loss, t_ll_e - t_ll_s))

        t_ll_s = time.time()
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(train_loader):
                if len(label) > 1:
                    for i in range(len(label)):
                        data_one = data[i]
                        label_one = label[i]
                        if args.cuda:
                            data_one, label_one = data_one.cuda(), label_one.cuda()
                        data_one, label_one = Variable(data_one), Variable(label_one)
                        train_loss += model.calculate_objective(data_one, label_one)[0] / len(label)
                        train_error += model.calculate_classification_error(data_one, label_one)[0] / len(label)
                else:
                    data_one = data[0]
                    label_one = label[0]
                    if args.cuda:
                        data_one, label_one = data_one.cuda(), label_one.cuda()
                    data_one, label_one = Variable(data_one), Variable(label_one)
                    train_loss += model.calculate_objective(data_one, label_one)[0] / len(label)
                    train_error += model.calculate_classification_error(data_one, label_one)[0] / len(label)
        t_ll_e = time.time()
        train_error /= len(train_loader)
        train_loss /= len(train_loader)
        print('\tTRAIN classification error value (time): {:.4f} ({:.2f}s)'.format(train_error, t_ll_e - t_ll_s))
        print('\tTRAIN log-likelihood value (time): {:.4f} ({:.2f}s)\n'.format(train_loss, t_ll_e - t_ll_s))

    if mode == 'test':
        return evaluate_loss, evaluate_error, train_loss, train_error
    else:
        return evaluate_loss, evaluate_error


def kfold_indices_warwick(N, k, seed=777):
    r = np.random.RandomState(seed)
    all_indices = np.arange(N, dtype=int)
    r.shuffle(all_indices)
    idx = [int(i) for i in np.floor(np.linspace(0, N, k + 1))]
    train_folds = []
    valid_folds = []
    for fold in range(k):
        valid_indices = all_indices[idx[fold]:idx[fold + 1]]
        valid_folds.append(valid_indices)
        train_fold = np.setdiff1d(all_indices, valid_indices)
        r.shuffle(train_fold)
        train_folds.append(train_fold)
    return train_folds, valid_folds
