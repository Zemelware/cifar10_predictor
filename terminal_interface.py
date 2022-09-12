import numpy as np
import classifier

print("Welcome! This is an AI trained on the CIFAR10 dataset. You can classify images in any of the following categories:")
print("airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck\n")

print("Make sure your image is a square for the best results!")
path = input("Please enter the path to the image you want to classify: ")

try:
    image = classifier.load_image(path)
except Exception as e:
    print(e)
    exit()
prediction = classifier.predict_image(image, return_probabilities=True)
prediction = sorted(prediction.items(), key=lambda x: x[1], reverse=True)

print("\n--------------------------------------------")
print("Here are the probabilities for each class:")
for i, (category, probability) in enumerate(prediction):
    if i == 0: print('\033[92m', end='')  # Print the prediction in green
    print(f"{category}: {'{:.2%}'.format(probability)}")
    if i == 0: print('\033[0m', end='')

# TODO: Put everything into a loop
# TODO: Add a GUI to make this more user-friendly
# TODO: Allow the user to specify the image as a command line argument
