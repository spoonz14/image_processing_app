from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import requests
import os
import random
import cv2

def contains_word(filename, word):
    # Convert both filename and word to lowercase for case-insensitive comparison
    return word.lower() in filename.lower()

processor = AutoImageProcessor.from_pretrained("NeuronZero/SkinCancerClassifier")
model = AutoModelForImageClassification.from_pretrained("NeuronZero/SkinCancerClassifier")

folder_path = 'D:\\PythonProjects\\image_processing_app\\TestCombined'

# Counter to track positives and negatives
positive_counter = 0
negative_counter = 0
# Counter to track number of images
counter = 0
# Counter to track false positives
false_p_counter = 0
# Counter to track false negatives
false_n_counter = 0
# Counter for true positives
true_p_counter = 0
# Counter for true negatives
true_n_counter = 0

filenames = os.listdir(folder_path)
random.shuffle(filenames)

for filename in filenames:
    file_path = os.path.join(folder_path, filename)
    # Load the image using OpenCV
    image = cv2.imread(file_path)

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    prediction = model.config.id2label[predicted_class_idx]
    print(f'Filename: {filename}')
    print(f'Prediction: {prediction}')
    if prediction == "NV":
        negative_counter += 1
        if contains_word(filename, "benign"):
            true_n_counter += 1
        if contains_word(filename, "malignant"):
            print("False negative!")
            false_n_counter += 1
    else:
        positive_counter += 1
        if contains_word(filename, "malignant"):
            true_p_counter += 1
        if contains_word(filename, "benign"):
            print("False positive!")
            false_p_counter += 1
    counter += 1

# Calculating accuracy
total_predictions = positive_counter + negative_counter
total_true_predictions = true_p_counter + true_n_counter
accuracy = total_true_predictions/total_predictions
accuracy = round(accuracy*100)

# Calculating precision
total_positives = true_p_counter + false_p_counter
precision = true_p_counter/total_positives
precision = round(precision*100)

# Calculating recall
total_true_positives = true_p_counter + false_n_counter
if true_p_counter and total_true_positives != 0:
    recall = true_p_counter/total_true_positives
    recall = recall*100
    recall = round(recall)
else:
    recall = "Recall error: divided by zero."

print(" ")
print(" ")
print('************ STATS *************')
print(f'Total predictions: {total_predictions}')
print(f'Total accurate predictions: {total_true_predictions}')
print(f'Number of positives: {positive_counter}')
print(f'Number of Negatives: {negative_counter}')
print(f'Number of False Negatives: {false_n_counter}')
print(f'Number of False Positives: {false_p_counter}')
print(f'Accuracy Rating: {accuracy}%')
print(f'Precision Rating: {precision}%')
print(f'Recall Rating: {recall}%')