import sys
import classifier

"""
import gradio as gr


def classify_image(image):
    img = image / 255
    return classifier.predict_image(img, return_probabilities=True)


# Gradio provides a web-based interface to test the model
gr.Interface(fn=classify_image, inputs=gr.Image(shape=(32, 32)), outputs="label").launch()
"""

if len(sys.argv) > 2:
    print("\033[91mYou've entered too many arguments. Please try again, with one argument as the filepath to the image "
          "you would like the AI to classify.\033[0m")
    sys.exit(1)

print("Welcome! This is an AI trained on the CIFAR10 dataset. The AI can classify images in any of the following categories:")
print("airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck\n")

print("Make sure your image is a square for the best results!")

first_time = True
while True:
    if len(sys.argv) == 1 or not first_time:
        path = input("\nPlease enter the path to the image you want to classify: ")
    elif len(sys.argv) == 2:
        path = sys.argv[1]

    try:
        image = classifier.load_image(path)
    except Exception as e:
        print(f"\n\033[91m{e}\033[0m")
        first_time = False
        continue
    prediction = classifier.predict_image(image, return_probabilities=True)
    prediction = sorted(prediction.items(), key=lambda x: x[1], reverse=True)

    print("\n--------------------------------------------")
    print("Here are the AI's predictions:")
    for i, (category, probability) in enumerate(prediction):
        if i == 0: print('\033[92m', end='')  # Print the prediction in green
        print(f"{category}: {'{:.2%}'.format(probability)}")
        if i == 0: print('\033[0m', end='')

    print("--------------------------------------------\n")
    choice = input("Do you want to classify another image? (y/n): ")
    if choice.lower() == 'n':
        break

    first_time = False
