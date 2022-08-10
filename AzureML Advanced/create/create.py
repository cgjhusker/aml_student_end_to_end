from tensorflow.keras import layers, models
import argparse
import mlflow
import logging

def create_model():

    mlflow.autolog()

    logging.basicConfig(level=logging.INFO)
    DATA_DIR = "model"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_output", type=str, help="Path of output model")
    args = parser.parse_args()
    logging.info("input parameters: %s", vars(args))

    

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    mlflow.keras.save_model(model, args.model_output)

if __name__ == "__main__":
    create_model()