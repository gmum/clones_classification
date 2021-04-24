import os
import sys

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.transforms import ColorJitter, RandomHorizontalFlip, RandomVerticalFlip, Resize

from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score

from data.patch_dataset import PatchesDataset
from utils.augmentation_utils import RandomRotate
from utils.bacteria_utils import read_config, get_loss
from utils.visualization import norm_cm

TB_DIR = 'tensorboard/'


def init_weights(m):
    torch.nn.init.xavier_uniform_(m.weight)


class FeaturePoolingCNN:
    def __init__(self, config):
        self.device = config['device'] if config['device'] is not None else torch.device('cuda')
        self.output_dim = config['output_dim']
        self.class_name = config['class_name']
        self.learning_rate = config['learning_rate']
        self.loss = get_loss(config['loss_function'])
        self.tb_writer = SummaryWriter(log_dir=os.path.join(TB_DIR, config['model_name']))
        self.model_name = config['model_name']
        self.model_path = config['model_path']
        self.model = self.prepare_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=config['lr_step'], gamma=config['lr_gamma'])
        self.dataloaders = {}
        self.dataiters = {}
        self.tb_writer.add_scalar(f'Learning_rate', self.learning_rate, 0)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def save_model(self, step):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        torch.save(self.model.state_dict(), os.path.join(self.model_path, f'{self.model_name}_{step}.pkl'))

    def prepare_model(self):
        model = models.resnet18(pretrained=True).eval().to(self.device)
        in_fc_dim = model.fc.in_features
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(in_fc_dim, self.output_dim)
        model.fc.apply(init_weights)
        model = model.cuda()
        return model

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
        y_hat = self.model(X).cpu().detach().numpy()
        del X
        return y_hat

    def init_data(self, train_dataloader, test_dataloader):
        self.dataloaders = {'train': train_dataloader, 'test': test_dataloader}
        self.dataiters = {'train': iter(train_dataloader), 'test': iter(test_dataloader)}

    def get_next_batch(self, phase):
        try:
            return next(self.dataiters[phase])
        except StopIteration:
            self.dataiters[phase] = iter(self.dataloaders[phase])
            return next(self.dataiters[phase])

    def train(self, train_data_loader,
              test_data_loader,
              total_steps,
              save_step):
        self.init_data(train_data_loader, test_data_loader)

        for step in tqdm(range(total_steps)):
            for phase in ['train', 'test']:
                X, y, _, _, _ = self.get_next_batch(phase)
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                inputs = X.to(self.device)
                labels = y.to(self.device)

                self.optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = self.loss(outputs, labels.long().reshape(outputs.shape[0]))

                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()

                y_vals = y.cpu().detach().numpy()
                preds_vals = preds.cpu().detach().numpy()
                cm = norm_cm(confusion_matrix(y_vals, preds_vals))
                acc = accuracy_score(y_vals, preds_vals)

                self.tb_writer.add_scalar(f'Loss/{phase}', loss.item(), step)
                self.tb_writer.add_scalar(f'Acc/{phase}', acc, step)
                for d in range(len(self.class_name)):
                    try:
                        self.tb_writer.add_scalar(f'{phase}/{self.class_name[d]}', cm[d, d], step)
                    except IndexError:
                        self.tb_writer.add_scalar(f'{phase}/{self.class_name[d]}', 0, step)

                if step % save_step == 0:
                    self.save_model(step)

            self.tb_writer.add_scalar(f'Learning_rate', get_lr(self.optimizer), step)
            self.scheduler.step()


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


if __name__ == '__main__':
    transforms = [ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                  RandomRotate(),
                  RandomHorizontalFlip(p=0.5),
                  RandomVerticalFlip(p=0.5)]
    config_file = sys.argv[1]
    config = read_config(config_file)
    model = FeaturePoolingCNN(config)
    train_dl = DataLoader(PatchesDataset(config['patches_filename'],
                                         config['labels_filename'],
                                         ['train'],
                                         config['split'],
                                         transforms=transforms),
                          batch_size=config['batch_size'], shuffle=True)
    test_dl = DataLoader(PatchesDataset(config['patches_filename'],
                                        config['labels_filename'],
                                        ['test'],
                                        config['split'],
                                        transforms=transforms),
                         batch_size=config['batch_size'], shuffle=True)
    model.train(train_dl, test_dl, 8000, 100)