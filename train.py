import cv2
import pandas as pd
import numpy as np
import os

from keras import backend as K
from keras import layers, models, optimizers, callbacks
from keras.preprocessing.image import ImageDataGenerator

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

import read_data

out_dir = 'output/' if os.path.exists('output/') else '/output/'
data_augmentation = False


def get_model(input_shape, n_classes):
    dropout_rate = 0.25

    model = models.Sequential()
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                            activation='relu', input_shape=input_shape,
                            kernel_initializer='glorot_normal',
                            bias_initializer='zeros'))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                            activation='relu', input_shape=input_shape,
                            kernel_initializer='glorot_normal',
                            bias_initializer='zeros'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Dropout(dropout_rate))

    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                            activation='relu',
                            kernel_initializer='glorot_normal',
                            bias_initializer='zeros'))
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                            activation='relu',
                            kernel_initializer='glorot_normal',
                            bias_initializer='zeros'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Dropout(dropout_rate))

    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same',
                            activation='relu',
                            kernel_initializer='glorot_normal',
                            bias_initializer='zeros'))
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same',
                            activation='relu',
                            kernel_initializer='glorot_normal',
                            bias_initializer='zeros'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Dropout(dropout_rate))

    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same',
                            activation='relu',
                            kernel_initializer='glorot_normal',
                            bias_initializer='zeros'))
    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same',
                            activation='relu',
                            kernel_initializer='glorot_normal',
                            bias_initializer='zeros'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Dropout(dropout_rate))

    model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same',
                            activation='relu',
                            kernel_initializer='glorot_normal',
                            bias_initializer='zeros'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Dropout(dropout_rate))

    model.add(layers.Flatten())

    model.add(layers.Dense(units=512, activation='relu',
                           kernel_initializer='glorot_normal',
                           bias_initializer='zeros'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(units=n_classes, activation='softmax',
                           kernel_initializer='glorot_normal',
                           bias_initializer='zeros'))

    optimizer = optimizers.Adam(lr=0.001, decay=1e-6)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()
    return model


def train():
    features, targets, n_classes = read_data.read_data()

    # preprocessing.Image.ImageDataGenerator(
    #
    # )

    print('total rows {}'.format(len(targets)))
    print('total classes {}'.format(n_classes))

    features = features.astype('float32')
    features /= 255

    (features_train, features_test,
     targets_train, targets_test) = train_test_split(features, targets,
                                                     test_size=0.3)

    model = get_model(features.shape[1:], n_classes)

    batch_size = 256
    epochs = 500

    log_dir = os.path.join(out_dir, 'image_gen')
    callback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,
                                     write_graph=True, write_grads=True,
                                     write_images=True)

    if data_augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=True,  # set input mean to 0 over the dataset
            samplewise_center=True,  # set each sample mean to 0
            featurewise_std_normalization=True,
            # divide inputs by std of the dataset
            samplewise_std_normalization=True,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,
            # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,
            # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,
            # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(features_train)

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(features_train, targets_train,
                                         batch_size=batch_size),
                            epochs=epochs, steps_per_epoch=len(features_test) / batch_size,
                            validation_data=(features_test, targets_test),
                            workers=4, callbacks=[callback], verbose=1)

    else:
        import ipdb; ipdb.set_trace()
        history = model.fit(features_train, targets_train,
                            validation_data=(features_test, targets_test),
                            batch_size=batch_size, epochs=epochs, verbose=2,
                            shuffle=True, callbacks=[callback])

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

