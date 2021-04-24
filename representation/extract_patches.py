import sys

import torch
from torch.utils.data import Dataset, DataLoader

from datetime import datetime

from utils.bacteria_utils import read_config
from representation.feature_pool_cnn import FeaturePoolingCNN
from data.patch_dataset import PatchesDataset


global value
value = 0


def get_least_but_one_output(module, input, output):
    global value
    value = output


class FeatureExtraction:

    def __init__(self, config):
        self.config = config
        self.device = config['device'] if config['device'] is not None else torch.device('cuda')
        self.save_name = config['save_name']
        self.prepare_cnn()

    def prepare_cnn(self):
        self.cnn_model = FeaturePoolingCNN(self.config)
        if 'load_model' in self.config.keys() and self.config['load_model']:
            self.cnn_model.load_model(self.config['save_model_path'])
        self.cnn_model.model.avgpool.register_forward_hook(get_least_but_one_output)

    def extract_features(self, patches_dataset, batch_size):
        batches = len(patches_dataset) // batch_size
        with torch.no_grad():
            features = {}
            dl = DataLoader(dataset=patches_dataset, batch_size=batch_size,
                            shuffle=False)

            for i, ds_sample in enumerate(dl):
                print(f'{datetime.now().strftime("%H:%M:%S")} Extracting batch: {i}/{batches}')
                images, _, ids, x, y = ds_sample
                image_batch = images.to(self.device)
                output = self.cnn_model.model.forward(image_batch)
                features_batch = value
                features_batch = features_batch.reshape(-1, features_batch.shape[1]).cpu()
                for i in range(len(ids)):
                    if ids[i] in features.keys():
                        features[ids[i]][f'{x[i]}_{y[i]}'] = features_batch[i]
                    else:
                        features[ids[i]] = {f'{x[i]}_{y[i]}': features_batch[i]}
            torch.save(features, self.save_name)


if __name__ == '__main__':
    config_file = sys.argv[1]
    config = read_config(config_file)
    extractor = FeatureExtraction(config)
    dataset = PatchesDataset(config['patches_filename'],
                             config['labels_filename'],
                             ['train', 'test', 'val', 'valid'],
                             split=config['split'],
                             transforms=None)
    extractor.extract_features(dataset, 64)