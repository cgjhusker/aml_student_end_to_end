
#! /usr/bin/env/python

import numpy as np
import os
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras import datasets
import numpy as np

if __name__ == "__main__":

    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data() 
    
    train_label_ohe = to_categorical(train_labels)
    test_label_ohe = to_categorical(test_labels)
    
    trainX = train_images.reshape((train_images.shape[0], 28, 28, 1))
    train_images_scaled = trainX / 255.0
    testX = test_images.reshape((test_images.shape[0], 28, 28, 1))
    test_images_scaled = testX / 255.0
    
    np.save(os.path.join("/opt/ml/processing/x_train", "x_train.npy"), train_images_scaled)
    #np.save(os.path.join("/opt/ml/processing/output", "x_test.npy"), test_images_scaled)
    np.save(os.path.join("/opt/ml/processing/y_train", "y_train.npy"), train_label_ohe)
    #np.save(os.path.join(raw_dir, "y_test.npy"), test_label_ohe)
