import os
import torch
import nrrd
import h5py
import numpy as np
import random

def read_h5(path):
    data = h5py.File(path, 'r')
    image = data['image'][:]
    label = data['label'][:]
    return image, label

class LAHeart(torch.utils.data.Dataset):
    def __init__(self, split='Training Set', label=True, transform=None):
        self.base_dir = f'/content/drive/MyDrive/newdataset/Dataset/{split}/'
        self.label = label
        self.split = split
        self.path_list = os.listdir(self.base_dir)
        self.transform = transform

    def __len__(self):
        if self.label and self.split == 'Training Set':
            return 32
        elif not self.label and self.split == 'Training Set':
            return (80-32)
        else:
            return 20

    def __getitem__(self, index):
        try:
            path = os.path.join(self.base_dir, self.path_list[index], "mri_norm2.h5")
        except Exception:
            path = os.path.join(self.base_dir, self.path_list[1], "mri_norm2.h5")
        image, label = read_h5(path)

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        return sample

# Augmentation Classes:
class RandomCrop:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] < self.output_size[0] or label.shape[1] < self.output_size[1] or label.shape[2] < self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        depth, height, width = image.shape
        target_d, target_h, target_w = self.output_size

        d_start = random.randint(0, depth - target_d)
        h_start = random.randint(0, height - target_h)
        w_start = random.randint(0, width - target_w)

        image = image[d_start:d_start+target_d, h_start:h_start+target_h, w_start:w_start+target_w]
        label = label[d_start:d_start+target_d, h_start:h_start+target_h, w_start:w_start+target_w]

        return {'image': image, 'label': label}


class RandomNoise:
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.random.normal(0, 0.1, image.shape)
        image = (image + noise).copy()  # ensure no negative stride
        return {'image': image, 'label': label}

class RandomRotFlip:
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        axis = random.randint(0, 2)
        k = random.randint(0, 3)
        image = np.rot90(image, k, axes=(axis, (axis+1)%3)).copy()
        label = np.rot90(label, k, axes=(axis, (axis+1)%3)).copy()
        if random.random() > 0.5:
            flip_axis = random.randint(0, 2)
            image = np.flip(image, axis=flip_axis).copy()
            label = np.flip(label, axis=flip_axis).copy()
        return {'image': image, 'label': label}

class ToTensor:
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = torch.from_numpy(image.copy()).float().unsqueeze(0)
        label = torch.from_numpy(label.copy()).long()
        return {'image': image, 'label': label}
