import pandas as pd
import torch

from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import Dataset

MEAN = [218.78237903, 194.69390888, 201.95128081]
STD = [2.61339548, 9.74849698, 4.84232922]


class PatchesDataset(Dataset):
    def __init__(self, patches_filename, labels_filename, modes, split, transforms):
        self.patches_df = pd.read_csv(patches_filename, index_col=0)
        self.labels_df = pd.read_csv(labels_filename, index_col=0)
        self.patches_df = self.patches_df[self.patches_df['image_id'].isin(
            self.labels_df[self.labels_df[split].isin(modes)]['image_id'])]

        if transforms is None:
            transforms = []
        transforms = [*transforms, Resize((250, 250)), ToTensor(), Normalize(mean=MEAN, std=STD)]

        self.transforms = Compose(transforms)

    def __len__(self):
        return len(self.patches_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        patch = self.patches_df.iloc[idx]
        patch_path = patch['path']
        image_id = patch['image_id']
        x, y = patch_path.split('/')[-1].replace('.png', '').split('_')
        img = Image.open(patch_path)
        img = self.transforms(img)
        label = self.labels_df[self.labels_df['image_id'] == image_id]['label'].values[0]
        return img, torch.Tensor([label]), image_id, x, y
