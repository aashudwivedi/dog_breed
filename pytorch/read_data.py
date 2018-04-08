import os
import torch
import numpy as np
import pandas as pd

from skimage import io
from torch.utils import data
from torchvision import transforms

local_input = os.path.join(os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.path.pardir)), 'input')
data_dir = local_input if os.path.exists(local_input) else '/input/'

float_dtype = torch.FloatTensor if torch.has_cudnn else torch.cuda.FloatTensor


class DogBreedDataSet(data.Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.names_frame = pd.read_csv(csv_file)
        self.classes = self.names_frame['breed']
        self.labels = self.classes.astype('category').cat.codes
        self.data_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.names_frame)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir,
                                  self.names_frame.iloc[idx][0] + '.jpg')
        image = io.imread(image_path)

        if self.transform:
            image = self.transform(image)

        label = torch.LongTensor([self.labels.iloc[idx].tolist()])
        return image, label


def get_dataset():
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

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
    loaders = {'train': train_loader, 'val': val_loader}
    sizes = {'train': len(train_idx), 'val': len(val_idx)}

    return loaders, sizes, len(np.unique(dataset.classes))







