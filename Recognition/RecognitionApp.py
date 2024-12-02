import cv2
import time
import numpy as np
from PIL import Image
import streamlit as st

# Function to preprocess the image for model prediction
def preprocess_image(img):
    # Resize the image to the target size (224x224 for model input)
    img = img.resize((224, 224))  # PIL resize
    img_array = np.array(img)  # Convert to numpy array
    img_array = img_array / 255.0  # Normalize to [0,1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1, 224, 224, 3)
    return img_array

def main():
    uploaded_file = ''
    # Set Streamlit page to be wide
    st.set_page_config(page_title="Melanoma Classifier", layout="wide")
    # Title of the app
    st.title("Melanoma Classifier")

    # Image upload functionality
    uploaded_file = st.file_uploader("Choose an Image to Process", type=["jpeg", "jpg", "png"])

    if uploaded_file is not None:
        st.write("File Uploaded Successfully")

        # Open the image using PIL
        img = Image.open(uploaded_file)


        img_resized = img.resize((600,400))
        # Show the uploaded image
        st.image(img_resized, caption="Uploaded Image",)

        # Preprocess the image for model prediction
        img_array = preprocess_image(img)

        import tensorflow as tf
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        # placeholder3 = st.empty()

        # with placeholder3:
        #     for i in range(6):
        #         placeholder3.text(f'Loading Tensorflow{"." * i}')
        #         time.sleep(1.25)
        #     placeholder3.text("Tensorflow loaded successfully.")
        #     time.sleep(1)

        # Load the pre-trained models
        model1 = tf.keras.models.load_model('D:/PythonProjects/image_processing_app/Melanoma-003.keras')
        st.write("Model 1 loaded.")
        model2 = AutoModelForImageClassification.from_pretrained("NeuronZero/SkinCancerClassifier")
        st.write("Model 2 loaded.")
        processor = AutoImageProcessor.from_pretrained("NeuronZero/SkinCancerClassifier")
        st.write("Processor loaded.")

        # Create a loading effect
        # placeholder1 = st.empty()

        # # Load the models with the loading effect
        # with placeholder1:
        #     # Display the effect
        #     for i in range(6):
        #         placeholder1.text(f'Loading Models{"." * i}')
        #         time.sleep(1.25)
        #     placeholder1.text("Models loaded successfuly.")
        #     time.sleep(1)

        # Create a loading effect
        placeholder2 = st.empty()

        # Run the prediction with the loading effect
        with placeholder2:
            # Display the effect
            for k in range(3):
                for i in range(6):
                    placeholder2.text(f'Working{"." * i}')
                    time.sleep(1.25)
            placeholder2.text("Done!")
            #time.sleep(1)

        # Predict the class
        prediction1 = model1.predict(img_array)

        # Interpret the result from model1
        if prediction1[0][0] > 0.3:  # threshold for melanoma
            st.write(f'Prediction 1: Melanoma')
            # Use model2 (transformers model) for cross-reference
            inputs = processor(images=img, return_tensors="pt")
            outputs = model2(**inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            prediction2 = model2.config.id2label[predicted_class_idx]
            if prediction2 != "NV":  # NV (Not Visible) means benign
                st.write(f'Prediction 2: Melanoma')
                st.write("Conclusion: Likely Melanoma.")
            else:
                st.write("Inconclusive, try another picture.")
        else:
            st.write("Prediction: Likely Benign.")

if __name__ == "__main__":
    main()
