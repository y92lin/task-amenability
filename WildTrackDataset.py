import os

import torch

import IQADataset
import pandas
import cv2
from torch.utils.data import Dataset

from brisque import brisque

class WildTrackDataset(Dataset):

    def __init__(self, dataset_file, config, status):
        self.gray_loader = IQADataset.gray_loader
        self.patch_size = config['patch_size']
        self.stride = config['stride']
        images = pandas.read_csv(dataset_file,header=None,names=["image_dir","species","class","image_filename","rating"])
        self.row_count = images.shape[0]

        # get rating
        self.mos = images["rating"].to_numpy()

        self.patches = ()
        self.features = []
        self.label = []

        for index, row in images.iterrows():
            print("Processing file number:" + str(index))
            file_path = os.path.join(row['image_dir'], row['image_filename'])
            im = self.gray_loader(file_path)
            im_features = cv2.imread(file_path)
            im_features = cv2.cvtColor(im_features, cv2.COLOR_BGR2RGB)
            features = brisque(im_features)
            patches = IQADataset.CropPatches(im, self.patch_size, self.stride)

            if status == 'train':
                self.patches = self.patches + patches
                for i in range(len(patches)):
                    self.label.append(self.mos[index])
                    self.features.append(features)
            else:
                self.patches = self.patches + (torch.stack(patches),)
                self.label.append(self.mos[index])
                self.features.append([])

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx], (torch.Tensor([self.label[idx]]), self.features[idx])