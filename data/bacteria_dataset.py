import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader

MEAN = [218.78237903, 194.69390888, 201.95128081]
STD = [2.61339548, 9.74849698, 4.84232922]


class BacteriaBag(Dataset):
    def __init__(self, df_labels, df_patches, features, split_column, curr_class=0, mode='train', shuffle=False):
        self.df_labels = df_labels
        self.df_patches = df_patches[df_patches['filter_std'] == 1]
        self.features = features
        self.split = split_column
        self.mode = mode
        self.shuffle = shuffle
        self.curr_class = curr_class
        self.bag_list, self.label_list = self.create_bag()

    def create_bag(self):
        bag_list = []
        labels_list = []
        curr_images = self.df_labels[self.df_labels[self.split] == self.mode]['image_id'].values
        for image_id in curr_images:
            curr_patches = self.df_patches[self.df_patches['image_id'] == image_id]['position'].values
            curr_features = self.features[image_id]
            curr_bag = [curr_features[x] for x in curr_patches]
            curr_label = self.df_labels[self.df_labels['image_id'] == image_id]['label'].values[0]
            if self.shuffle:
                random.shuffle(curr_bag)
            bag_list.append(torch.stack(curr_bag))
            labels_list.append(curr_label)
        return bag_list, labels_list

    def __len__(self):
        return len(self.bag_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.bag_list[idx], self.label_list[idx]


class BacteriaBagWithCoord(Dataset):
    def __init__(self, df_labels, df_patches, features, split_column, curr_class=0, mode='train', shuffle=False):
        self.df_labels = df_labels
        self.df_patches = df_patches[df_patches['filter_std'] == 1]
        self.features = features
        self.split = split_column
        self.mode = mode
        self.shuffle = shuffle
        self.curr_class = curr_class
        self.bag_list, self.label_list, self.coordinates, self.paths = self.create_bag()

    def create_bag(self):
        bag_list = []
        labels_list = []
        coords_list = []
        paths_list = []
        curr_images = self.df_labels[self.df_labels[self.split] == self.mode]['image_id'].values
        for image_id in curr_images:
            curr_patches = self.df_patches[self.df_patches['image_id'] == image_id]['position'].values
            curr_patches_paths = self.df_patches[self.df_patches['image_id'] == image_id]['path'].values
            curr_features = self.features[image_id]
            curr_bag = [curr_features[x] for x in curr_patches]
            curr_label = self.df_labels[self.df_labels['image_id'] == image_id]['label'].values[0]
            if self.shuffle:
                random.shuffle(curr_bag)
            bag_list.append(torch.stack(curr_bag))
            labels_list.append(curr_label)
            curr_patches_xy = []
            for coord in curr_patches:
                x, y = [int(a) for a in coord.split('_')]
                curr_patches_xy.append(np.array([x, y]))
            coords_list.append(np.array(curr_patches_xy))
            paths_list.append(curr_patches_paths)
        return bag_list, labels_list, np.array(coords_list), paths_list

    def __len__(self):
        return len(self.bag_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.bag_list[idx], self.label_list[idx], self.coordinates[idx], self.paths[idx]

    def get_paths(self, idx):
        return self.paths[idx]


def image_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]


def image_collate2(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    coordinates = [item[2] for item in batch]
    paths = [item[3] for item in batch]
    return [data, target, coordinates, paths]
