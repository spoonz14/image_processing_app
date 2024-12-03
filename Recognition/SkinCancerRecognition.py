import os
import cv2
import random
import numpy as np
import tensorflow as tf
from transformers import pipeline


# Load the pre-trained model
model = tf.keras.models.load_model('D:\PythonProjects\image_processing_app\skin_cancer_model.keras')

def preprocess_image(img_path):
    # Load the image using OpenCV
    img = cv2.imread(img_path)
    # Resize the image to the target size
    img = cv2.resize(img, (224, 224))
    # Normalize the image
    img_array = img / 255.0
    # Reshape to add batch dimension
    img_array = np.expand_dims(img_array, axis=0)  # This adds the batch dimension
    return img_array

def contains_word(filename, word):
    # Convert both filename and word to lowercase for case-insensitive comparison
    return word.lower() in filename.lower()

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

#image_path = "D:\PythonProjects\image_processing_app\Test1s\malignant0.jpg"
# Swapping to folder to iterate through many images
folder_path = 'D:\\PythonProjects\\image_processing_app\\TestCombined'

# Setting folder as a list directory
filenames = os.listdir(folder_path)
random.shuffle(filenames)

for filename in filenames:
    file_path = os.path.join(folder_path, filename)
    processed_image = preprocess_image(file_path)

    
    prediction = model.predict(processed_image)

    print(f'Filename: {filename}')
    print(f'Prediction: {prediction}')
    # Check the shape of the prediction
    #print(f'Prediction shape: {prediction.shape}')

    # Extract the predicted class (1 = benign, 2 = malignant)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_probability = np.max(prediction)
    print(f'Predicted class: {predicted_class}')
    print(f'Predicted probability: {predicted_probability}')

    if predicted_class == 1:
        if predicted_probability < 0.65:
            print(f'Likely Benign with {predicted_probability*100:.2f}% confidence.')
            negative_counter = negative_counter + 1
            word_to_check = "malignant"
            if contains_word(filename, word_to_check):
                print("False negative!")
                false_n_counter = false_n_counter + 1
            else:
                true_n_counter = true_n_counter + 1
        else:
            print(f'Likely Malignant with {predicted_probability*100:.2f}% confidence.')
            positive_counter = positive_counter + 1
            word_to_check = "benign"
            if contains_word(filename, word_to_check):
                print("False positive!")
                false_p_counter = false_p_counter + 1
            else:
                true_p_counter = true_p_counter + 1
    
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
if true_p_counter and total_true_positives == 0:
    recall = "Recall error: divided by zero."
elif false_n_counter == 0:
    recall = 100
else:
    recall = true_p_counter/total_true_positives
    recall = recall*100
    recall = round(recall)

print(" ")
print(" ")
print('************ STATS **************')
print(f'Total predictions: {total_predictions}')
print(f'Total accurate predictions: {total_true_predictions}')
print(f'Number of positives: {positive_counter}')
print(f'Number of Negatives: {negative_counter}')
print(f'Number of False Negatives: {false_n_counter}')
print(f'Number of False Positives: {false_p_counter}')
print(f'Accuracy Rating: {accuracy}%')
print(f'Precision Rating: {precision}%')
print(f'Recall Rating: {recall}%')