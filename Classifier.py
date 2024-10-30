from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import requests
import os
import random
import cv2

def preprocess_image(img_path):
    
    # Resize the image to the target size
    img = cv2.resize(img, (224, 224))
    # Normalize the image
    img_array = img / 255.0
    # Reshape to add batch dimension
    #img_array = np.expand_dims(img_array, axis=0)  # This adds the batch dimension
    return img_array

def contains_word(filename, word):
    # Convert both filename and word to lowercase for case-insensitive comparison
    return word.lower() in filename.lower()

processor = AutoImageProcessor.from_pretrained("NeuronZero/SkinCancerClassifier")
model = AutoModelForImageClassification.from_pretrained("NeuronZero/SkinCancerClassifier")

folder_path = 'D:\\PythonProjects\\image_processing_app\\ImageProcessingApp\\Test0s'

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
    print("Predicted class:", model.config.id2label[predicted_class_idx])