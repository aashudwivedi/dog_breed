import cv2
import pandas as pd
import numpy as np
import os

from keras import backend as K
from keras import preprocessing, layers, models, optimizers, callbacks
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
K.set_image_dim_ordering("tf")

data_dir = 'input/' if os.path.exists('input/') else '/input/'
out_dir = 'output/' if os.path.exists('output/') else '/output/'


def read_data():
    labels = pd.read_csv(data_dir + 'labels.csv')

    # only top 20 classes
    # idx = np.zeros(labels.shape[0], np.bool)
    # for breed in labels['breed'].value_counts()[:20].index.values:
    #     idx = idx | (labels['breed'] == breed)

    # labels = labels[idx]

    n_classes = len(np.unique(labels['breed']))
    path = data_dir + 'train/{}.jpg'
    width = 128
    height = 128

    images = []
    for filename, _ in labels.values:
        image = cv2.imread(path.format(filename))
        resized = cv2.resize(image, (width, height)) / 255
        images.append(resized)

    features = np.array(images, np.float)
    features -= features.mean(axis=0)
    targets = pd.get_dummies(labels['breed'], sparse=True).values
    return features, targets, n_classes


def get_model(input_shape, n_classes):
    dropout_rate = 0.25

    model = models.Sequential()
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                            activation='relu', input_shape=input_shape,
                            kernel_initializer='glorot_normal',
                            bias_initializer='zeros'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 2), padding="same"))
    model.add(layers.Dropout(dropout_rate))

    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                            activation='relu',
                            kernel_initializer='glorot_normal',
                            bias_initializer='zeros'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 2), padding="same"))
    model.add(layers.Dropout(dropout_rate))

    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                            activation='relu',
                            kernel_initializer='glorot_normal',
                            bias_initializer='zeros'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 2), padding="same"))
    model.add(layers.Dropout(dropout_rate))

    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                            activation='relu',
                            kernel_initializer='glorot_normal',
                            bias_initializer='zeros'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 2), padding="same"))
    model.add(layers.Dropout(dropout_rate))

    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                            activation='relu',
                            kernel_initializer='glorot_normal',
                            bias_initializer='zeros'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 2), padding="same"))
    model.add(layers.Dropout(dropout_rate))

    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                            activation='relu',
                            kernel_initializer='glorot_normal',
                            bias_initializer='zeros'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 2), padding="same"))
    model.add(layers.Dropout(dropout_rate))

    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                            activation='relu',
                            kernel_initializer='glorot_normal',
                            bias_initializer='zeros'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 2), padding="same"))
    model.add(layers.Dropout(dropout_rate))

    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                            activation='relu',
                            kernel_initializer='glorot_normal',
                            bias_initializer='zeros'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 2), padding="same"))
    model.add(layers.Dropout(dropout_rate))

    model.add(layers.Flatten())

    model.add(layers.Dense(units=n_classes, activation='relu',
                           kernel_initializer='glorot_normal',
                           bias_initializer='zeros'))

    model.add(layers.Dense(units=n_classes, activation='softmax',
                           kernel_initializer='glorot_normal',
                           bias_initializer='zeros'))

    optimizer = optimizers.Adam(lr=0.001, decay=1e-6)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()
    return model


def train():
    features, targets, n_classes = read_data()

    print('total rows {}'.format(len(targets)))
    print('total classes {}'.format(n_classes))

    (features_train, features_test,
     targets_train, targets_test) = train_test_split(features, targets,
                                                     test_size=0.3)

    model = get_model(features.shape[1:], n_classes)

    batch_size = 256
    epochs = 500

    log_dir = out_dir
    callback = callbacks.TensorBoard(log_dir='/output/', histogram_freq=0,
                                     write_graph=True, write_grads=True,
                                     write_images=True)

    history = model.fit(features_train, targets_train,
                        validation_data=(features_test, targets_test),
                        batch_size=batch_size, epochs=epochs, verbose=2,
                        callbacks=[callback])

    return history, model


def save_model(model):
    model_json = model.to_json()
    with open(os.path.join(out_dir, 'model.json'), 'w') as f:
        f.write(model_json)
    model.save_weights(os.path.join(out_dir, 'model.h5'))


def plot(history):
    for key in history.history.keys():
        plt.plot(history.history[key])
        plt.savefig(os.path.join(out_dir,'plots/{}.png'.format(key)))


history, model = train()
save_model(model)
# plot(history)

