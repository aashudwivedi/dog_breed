import os
import torch
import numpy as np
import pandas as pd

from PIL import Image
from skimage import io
from torch.utils import data
from torchvision import transforms, datasets

data_dir = '../input/' if os.path.exists('../input/') else '/input/'


class DogBreedDataSet(data.Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.names_frame = pd.read_csv(csv_file)
        self.data_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.names_frame)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir,
                                  self.names_frame.iloc[idx, 0] + '.jpg')
        image = io.imread(image_path)
        # image = Image.open(image_path)

        # print('before transform image size = {}'.format(image.shape))
        # print('idx =', idx)
        if self.transform:
            image = self.transform(image)

            # print('after transform image size = {}'.format(image.shape))

        breed = self.names_frame.iloc[idx, 1]

        return {'image': image, 'breed': breed}


def get_dataset():
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor()])

    return DogBreedDataSet(
        os.path.join(data_dir, 'labels.csv'),
        os.path.join(data_dir, 'train'), transform=data_transforms)


def get_loader():
    dog_dataset = get_dataset()
    data_loader = data.DataLoader(
        dog_dataset, batch_size=4, shuffle=True, num_workers=4)
    return data_loader


def get_train_val_loader(validation_size=0.3, shuffle=True):
    dataset = get_dataset()

    idxes = np.arange(len(dataset))
    if shuffle:
        np.random.shuffle(idxes)

    split = np.floor(len(dataset) * validation_size).astype('int')
    train_idx = idxes[split:]
    val_idx = idxes[:split]

    train_sampler = data.sampler.SubsetRandomSampler(train_idx)
    test_sampler = data.sampler.SubsetRandomSampler(val_idx)

    train_loader = data.DataLoader(dataset=dataset,
                                   batch_size=4,
                                   sampler=train_sampler,
                                   num_workers=4)

    val_loader = data.DataLoader(dataset=dataset,
                                 batch_size=4,
                                 sampler=test_sampler,
                                 num_workers=4)
    return train_loader, val_loader







