import cv2
import pandas as pd
import numpy as np


from keras import preprocessing, layers, models

labels = pd.read('data/labels.csv')

labels.head()


