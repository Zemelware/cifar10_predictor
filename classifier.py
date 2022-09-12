from PIL import Image
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
X_train = X_train / 255
X_test = X_test / 255
y_train = y_train.reshape(-1, )  # Convert to 1-dimensional array

classes = ["airplane", "automobile", "bird", "cat",
           "deer", "dog", "frog", "horse", "ship", "truck"]


def predict_image(img, return_probabilities=False):
    predict_img = np.expand_dims(img, axis=0)
    probabilities = cnn.predict(predict_img)
    prediction = classes[np.argmax(probabilities[0])]
    if not return_probabilities:
        return prediction
    else:
        return probabilities.tolist()[0]


def plot_sample(x, y, index, show_prediction=False):
    if show_prediction:
        prediction = predict_image(x[index])
        plt.title(f"Prediction: {prediction}")

    plt.imshow(x[index])
    plt.xlabel(f"Label: {classes[y[index]]}")
    plt.show()


def load_image(path):
    img = Image.open(path).resize((32, 32), Image.ANTIALIAS)
    img = np.array(img)
    img = img / 255
    return img


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
