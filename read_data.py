import cv2
import os
import pandas as pd
import numpy as np

data_dir = 'input/' if os.path.exists('input/') else '/input/'


def read_data():
    labels = pd.read_csv(data_dir + 'labels.csv')

    # only top 20 classes
    idx = np.zeros(labels.shape[0], np.bool)
    for breed in labels['breed'].value_counts()[:10].index.values:
        idx = idx | (labels['breed'] == breed)

    # labels = labels[idx]

    n_classes = len(np.unique(labels['breed']))
    path = data_dir + 'train/{}.jpg'
    width = 32
    height = 32

    images = []
    for filename, _ in labels.values:
        image = cv2.imread(path.format(filename))
        resized = cv2.resize(image, (width, height))
        images.append(resized)

    features = np.array(images, np.float32)
    # features -= features.mean(axis=0)
    targets = pd.get_dummies(labels['breed'], sparse=True).values
    return features, targets, n_classes