import numpy as np
import classifier

print("Welcome! This is an AI trained on the CIFAR10 dataset. You can classify images in any of the following categories:")
print("airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck\n")

path = input("Please enter the path to the image you want to classify: ")

try:
    image = classifier.load_image(path)
except Exception as e:
    print(e)
    exit()
prediction = classifier.predict_image(image, return_probabilities=True)
prediction = sorted(prediction, reverse=True)

print("\n--------------------------------------------")
print("Here are the probabilities for each class:")
for i in range(len(prediction)):
    print(f"{classifier.classes[i]}: {'{:.2%}'.format(prediction[i])}")

print("The AI predicts that this image is a(n)", classifier.classes[np.argmax(prediction)] + '.')
