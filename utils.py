import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms
import numpy as np
import os

class Rotate3D:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, image, label):
        angle = np.random.choice(self.angles)

        image = np.rot90(image, k=angle//90, axes=(1, 2))
        label = np.rot90(label, k=angle//90, axes=(1, 2))

        return image, label

class Flip3D:
    def __init__(self):
        pass

    def __call__(self, image, label):
        image = np.flipud(image)
        label = np.flipud(label)

        return image, label

class RandomNoise:
    def __init__(self, noise_level=0.05):
        self.noise_level = noise_level

    def __call__(self, image, label):
        noise = np.random.normal(0, self.noise_level, image.shape)
        return image + noise, label

class SeismicDataset(Dataset):
    def __init__(self, dpath, fpath, dim=(128, 128, 128), augment_factor=0):
        self.dim = dim
        self.dpath = dpath
        self.fpath = fpath
        self.data = os.listdir(dpath)
        self.fault = os.listdir(fpath)
        self.augment_factor = augment_factor

    def __len__(self):
        return len(self.data) * (self.augment_factor + 1)

    def __getitem__(self, idx):
        original_idx = idx // (self.augment_factor + 1)
        augment_idx = idx % (self.augment_factor + 1)

        gx = np.fromfile(os.path.join(self.dpath, self.data[original_idx]), dtype=np.float32)
        fx = np.fromfile(os.path.join(self.fpath, self.fault[original_idx]), dtype=np.float32)

        gx = np.reshape(gx, self.dim)
        fx = np.reshape(fx, self.dim)

        gx = (gx - np.mean(gx)) / np.std(gx)

        gx = np.transpose(gx)
        fx = np.transpose(fx)

        if augment_idx > 0:
            self.transform = get_transform(train=True)
        else:
            self.transform = get_transform(train=False)
            
        X, Y = self.transform(gx, fx)
        X = torch.from_numpy(X.copy()).float().unsqueeze(0)
        Y = torch.from_numpy(Y.copy()).float().unsqueeze(0)

        return X, Y
    
def get_transform(train):
	if train:
		return transforms.Compose([
			Rotate3D(),
            Flip3D(),
            RandomNoise(noise_level=0.05),
			transforms.ToDtype(torch.float32, scale=True),
		])
	else:
		return transforms.Compose([
			transforms.ToDtype(torch.float32, scale=True),
		])
