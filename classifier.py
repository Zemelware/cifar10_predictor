import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress annoying TensorFlow warnings


def preprocess_data():
    (X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
    X_train = X_train / 255
    X_test = X_test / 255
    y_train = y_train.reshape(-1, )  # Convert to 1-dimensional array
    return X_train, y_train, X_test, y_test


def predict_image(img, return_probabilities=False):
    predict_img = np.expand_dims(img, axis=0)
    probabilities = model.predict(predict_img).tolist()
    prediction = classes[np.argmax(probabilities[0])]
    if not return_probabilities:
        return prediction
    else:
        # Return a dictionary of class: probability pairs
        return {classes[i]: probabilities[0][i] for i in range(len(classes))}


def plot_sample(x, y, index, show_prediction=False):
    if show_prediction:
        prediction = predict_image(x[index])
        plt.title(f"Prediction: {prediction}")

    plt.imshow(x[index])
    plt.xlabel(f"Label: {classes[y[index]]}")
    plt.show()


def plot_accuracy(model_history):
    plt.plot(model_history.history['accuracy'], label='Train')
    plt.plot(model_history.history["val_accuracy"], label="Validation")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.show()


def plot_loss(model_history):
    plt.plot(model_history.history['loss'], label='Training')
    plt.plot(model_history.history["val_loss"], label="Validation")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.show()


def load_image(path):
    try:
        img = Image.open(path).convert('RGB').resize((32, 32), Image.ANTIALIAS)
    except UnidentifiedImageError:
        raise Exception("The image could not be loaded. Are you sure this is an image file?")

    img = np.array(img)
    img = img / 255
    return img


def create_model(learning_rate, num_classes):
    with tf.device('/CPU:0'):  # Fix M1 TensorFlow Metal bug
        data_augmentation = models.Sequential([
            layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(32, 32, 3)),
            layers.experimental.preprocessing.RandomWidth(0.1),
            layers.experimental.preprocessing.RandomHeight(0.1),
        ])

    cnn = models.Sequential([
        data_augmentation,

        # CNN
        layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        # Dense
        # layers.Flatten(),
        layers.GlobalMaxPool2D(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])

    cnn.compile(optimizer=optimizers.Adam(learning_rate),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])

    return cnn


classes = ["airplane", "automobile", "bird", "cat",
           "deer", "dog", "frog", "horse", "ship", "truck"]

LEARNING_RATE = 0.001
EPOCHS = 1000
BATCH_SIZE = 200

weights_path = "best_weights/best_weights.ckpt"
weights_dir = os.path.dirname(weights_path)

model = create_model(LEARNING_RATE, len(classes))
if os.path.exists(weights_dir):
    model.load_weights(weights_path)

if __name__ == "__main__":
    # Create a callback that saves the model's best weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=weights_path,
                                                     monitor='val_accuracy',
                                                     save_weights_only=True,
                                                     save_best_only=True,
                                                     mode='max')

    # Create an early stopping callback
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=10,
                                                   verbose=1,
                                                   mode='min')

    X_train, y_train, X_test, y_test = preprocess_data()

    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        validation_data=(X_test, y_test), shuffle=True, callbacks=[cp_callback, es_callback])
    plot_accuracy(history)
    plot_loss(history)
    model.evaluate(X_test, y_test)
