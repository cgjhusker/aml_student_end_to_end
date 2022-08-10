from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras import datasets
import argparse
from tensorflow.keras.models import load_model
import mlflow

def train_model():
    mlflow.autolog()
    DATA_DIR = "model"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_in", type=str, help="Path of output model")
    parser.add_argument("--model_out", dest="model_dir", default=DATA_DIR)

    args = parser.parse_args()
    model_in = mlflow.keras.load_model(args.model_in)
        
    fashion_mnist = tf.keras.datasets.fashion_mnist
    print("here1")
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data() 
    print("here2")

    trainX = train_images.reshape((train_images.shape[0], 28, 28, 1))
    testX = test_images.reshape((test_images.shape[0], 28, 28, 1))
    train_images_scaled = trainX / 255.0
    test_images_scaled = testX / 255.0
    train_label_ohe = to_categorical(train_labels)
    test_label_ohe = to_categorical(test_labels)

    model_in.fit(train_images_scaled, train_label_ohe, validation_split=0.15, epochs=5)

if __name__ == "__main__":
    print("hi")

    train_model()
