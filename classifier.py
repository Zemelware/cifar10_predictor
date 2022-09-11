import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
X_train = X_train / 255
X_test = X_test / 255
y_train = y_train.reshape(-1, )  # Convert to 1-dimensional array

classes = ["airplane", "automobile", "bird", "cat",
           "deer", "dog", "frog", "horse", "ship", "truck"]


def plot_sample(x, y, index, show_prediction=False):
    if show_prediction:
        predict_img = np.expand_dims(X_train[index], axis=0)
        prediction = cnn.predict(predict_img)
        class_prediction = classes[np.argmax(prediction[0])]
        plt.title(f"Prediction: {class_prediction}")

    plt.imshow(x[index])
    plt.xlabel(f"Label: {classes[y[index]]}")
    plt.show()


cnn = models.Sequential([
    # CNN
    layers.Conv2D(filters=32, kernel_size=(3, 3),
                  activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(filters=32, kernel_size=(3, 3),
                  activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    # Dense
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation="softmax")
])

cnn.compile(optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"])

# Load the model
cnn = models.load_model("cifar10_cnn.h5")

# cnn.fit(X_train, y_train, epochs=10)
# cnn.evaluate(X_test, y_test)

# Save the model
# cnn.save("cifar10_cnn.h5")
