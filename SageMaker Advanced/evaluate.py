
import os
import json
import subprocess
import sys
import numpy as np
import pathlib
import tarfile


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


if __name__ == "__main__":

    install("tensorflow==2.4.1")
    import tensorflow as tf
    from tensorflow.keras import layers, models
    
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(layers.Dense(10, activation='softmax'))

    test_path1 = "/opt/ml/processing/x_train/"
    test_path2 = "/opt/ml/processing/y_train/"
    x_train = np.load(os.path.join(test_path1, "x_train.npy"))
    y_train = np.load(os.path.join(test_path2, "y_train.npy"))
    
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    model.fit(x_train, y_train, validation_split=0.15, epochs=5)
