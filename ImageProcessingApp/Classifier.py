import os
import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Load the processor and model
processor = AutoImageProcessor.from_pretrained("gianlab/swin-tiny-patch4-window7-224-finetuned-skin-cancer")
model = AutoModelForImageClassification.from_pretrained("gianlab/swin-tiny-patch4-window7-224-finetuned-skin-cancer")

# Set the model to evaluation mode
model.eval()

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

folder_path = 'D:\\PythonProjects\\image_processing_app\\ImageProcessingApp\\Test0s'

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    processed_image = preprocess_image(file_path)

    # Convert to tensor and ensure it's of type float32
    input_tensor = torch.tensor(processed_image, dtype=torch.float32).permute(0, 3, 1, 2)  # Change from (N, H, W, C) to (N, C, H, W)
    
    # Make predictions
    with torch.no_grad():  # Disable gradient calculation
        outputs = model(input_tensor)
    
    # Get the predicted class (argmax)
    predictions = torch.argmax(outputs.logits, dim=1)

    if predictions.item() >= 4:
        print(f'Prediction: {predictions.item()}; likely Cancer.')
    else:
        print(f'Prediction: {predictions.item()}; not Cancer.')

    # Display the image
    image_to_show = (processed_image[0] * 255).astype(np.uint8)  # Get the first image and convert
    cv2.imshow('Processed Image', image_to_show)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()  # Close the window

    
