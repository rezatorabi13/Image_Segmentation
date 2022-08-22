import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K


def dice_coef(y_true, y_pred):   #mask1 , mask2
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true) + K.sum(y_pred) + K.epsilon())
    

