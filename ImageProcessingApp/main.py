import numpy as np
import streamlit as st
import cv2
from PIL import Image

def apply_filters(image, filter_type): #Takes filter type as input and applies to image
    if filter_type == "Grayscale":
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) #Converts to greyscale
    elif filter_type == "Smoothing":
        return cv2.GaussianBlur(image, (15, 15), 0)
    else:
        return image

def main():
    st.title("Image Processor")

    uploaded_file = st.file_uploader("Choose an Image to Process", type=["jpeg", "jpg", "png"]) #accepts JPEG, PNG, and JPG image types for processing

    if uploaded_file is not None: #Checks if file is uploaded
        st.write("File Uploaded")

        img = Image.open(uploaded_file) #Uses PIL library to open image
        img_array = np.array(img) #Image converted to NumPy array so it can be processed by OpenCV, array represents pixel values (RBG format)

        st.image(img, caption="Uploaded Image", use_column_width=True)

        filter_type = st.selectbox("Choose a Filter", ["None", "Grayscale", "Smoothing"])

        processed_img = apply_filters(img_array, filter_type)

        if filter_type == "Grayscale":
            st.image(processed_img, caption="Processed Image", use_column_width=True, channels="GRAY") #If Greyscale filter chosen
        else:
            st.image(processed_img, caption="Processed Image", use_column_width=True) #If Smoothing or none - both are in RBG 

if __name__ == "__main__":
    main()
