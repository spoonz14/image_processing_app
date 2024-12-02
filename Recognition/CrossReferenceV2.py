import os
import tensorflow as tf
import cv2
import random
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification

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

# Load the pre-trained model
model1 = tf.keras.models.load_model('D:\PythonProjects\image_processing_app\Melanoma-003.keras')
model2 = AutoModelForImageClassification.from_pretrained("NeuronZero/SkinCancerClassifier")
model3 = tf.keras.models.load_model('D:\PythonProjects\image_processing_app\skin_cancer_model.keras')

processor = AutoImageProcessor.from_pretrained("NeuronZero/SkinCancerClassifier")

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
    # Counter to track model predictions
    positive_prediction_counter = 0
    file_path = os.path.join(folder_path, filename)
    processed_image = preprocess_image(file_path)
    image = cv2.imread(file_path)

    # Convert to uint8 for display
    #image_to_show = (processed_image[0] * 255).astype(np.uint8)  # Get the first image and convert
    
    print("First Prediction:")
    # Predict the class
    prediction1 = model1.predict(processed_image)
    print(f'Filename: {filename}')
    print(f'Prediction: {prediction1}')
    # Check the shape of the prediction
    print(f'Prediction shape: {prediction1.shape}')

    # Interpret the result
    if prediction1[0] > 0.3:
        # Increment the prediction counter
        positive_prediction_counter += 1
        print(f'Positve Predictions: {positive_prediction_counter}')

    print("Second prediction:")
    inputs = processor(images=image, return_tensors="pt")
    outputs = model2(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    prediction2 = model2.config.id2label[predicted_class_idx]
    print(f'Cross Reference Prediction: {prediction2}')
    if prediction2 != "NV":
        positive_prediction_counter +=1
        print(f'Positve Predictions: {positive_prediction_counter}')

    if positive_prediction_counter == 2:
        print(f'Filename: {filename}')
        print('The lesion is classified as Melanoma.')
        positive_counter = positive_counter + 1
        word_to_check = 'benign'
        if contains_word(filename, word_to_check):
            false_p_counter = false_p_counter + 1
            print('(False Positive!)')
        else:
            true_p_counter = true_p_counter +1

    elif positive_prediction_counter == 1:
        print("Third Prediction:")
        prediction3 = model3.predict(processed_image)

        # Extract the predicted class (0 = benign, 1 = malignant)
        predicted_class = np.argmax(prediction3, axis=1)
        predicted_probability = np.max(prediction3)
        print(f'Predicted class: {predicted_class}')
        print(f'Predicted probability: {predicted_probability}')

        if predicted_class == 1:
            if predicted_probability < 0.63:
                print(f'Likely Benign.')
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

    elif positive_prediction_counter == 0:
        print('The lesion is classified as Not Melanoma.')
        negative_counter = negative_counter + 1
        word_to_check = 'malignant'
        if contains_word(filename, word_to_check):
            false_n_counter = false_n_counter + 1
            print('(False Negative!)')
        else:
            true_n_counter = true_n_counter + 1

    counter = counter + 1

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