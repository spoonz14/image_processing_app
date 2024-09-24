import numpy as np
import streamlit as st
import cv2
from PIL import Image

def choose_smoothing(smoothing_choice):
    # Change filters depending on choice input
    if smoothing_choice == "GaussianBlur":
        return "GaussianBlur"
    elif smoothing_choice == "Median":
        return "Median"

def apply_filters(image, filter_type):
    # Ensure the input image is in the correct format for the selected filter
    if filter_type == "Grayscale":
        if len(image.shape) == 3:  # Only convert if it's a color image
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif filter_type == "GaussianBlur":
        return cv2.GaussianBlur(image, (15, 15), 0)
    elif filter_type == "Median":
        return cv2.medianBlur(image, 3)
    elif filter_type == "Equalize":
        if len(image.shape) == 3:  # Convert to grayscale if it's a color image
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            return cv2.equalizeHist(gray_image)
        return cv2.equalizeHist(image)
    elif filter_type == "CLAHE":
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image
        clahe = cv2.createCLAHE(clipLimit=5)
        return clahe.apply(gray_image)
    elif filter_type == "Binary":
        _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return cv2.bitwise_not(binary_image)  # Invert the binary image
    return image  # Return the original image if filter type is None

def main():
    # Initialization
    if 'saved_img' not in st.session_state:
        st.session_state.saved_img = None
    if 'processed_img' not in st.session_state:
        st.session_state.processed_img = None

    st.title("Image Processor")

    uploaded_file = st.file_uploader("Choose an Image to Process", type=["jpeg", "jpg", "png"])

    if uploaded_file is not None:
        st.write("File Uploaded")
        img = Image.open(uploaded_file)
        img_array = np.array(img)

        # Display the uploaded image only once
        if st.session_state.saved_img is None:
            st.image(img, caption="Uploaded Image", use_column_width=True)
            st.session_state.saved_img = img_array
            st.session_state.processed_img = img_array.copy()  # Initialize processed image

        elif st.session_state.saved_img is not None:
            # Display the original uploaded image
            st.image(st.session_state.saved_img, caption="Original Image", use_column_width=True)

        smoothing_choice = st.selectbox("Choose smoothing type", ["GaussianBlur", "Median"])

        filter_type = st.selectbox("Choose a Filter", ["None", "Grayscale", "Smoothing", "Equalize", "CLAHE", "Binary"], key="filter_choice")

        if st.button("Apply Filter"):
            if filter_type == "Smoothing":
                chosen_smoothing = choose_smoothing(smoothing_choice)
                st.session_state.processed_img = apply_filters(st.session_state.saved_img, chosen_smoothing)
            elif filter_type != "None":
                st.session_state.processed_img = apply_filters(st.session_state.saved_img, filter_type)

            # Update saved_img to the latest processed image only if filter was applied
            if filter_type != "None":
                st.session_state.saved_img = st.session_state.processed_img.copy()

            # Display the processed image
            if filter_type == "Grayscale":
                st.image(st.session_state.processed_img, caption="Processed Image (Grayscale)", use_column_width=True, channels="GRAY")
            elif filter_type == "Smoothing":
                st.image(st.session_state.processed_img, caption="Processed Image (Smoothing)", use_column_width=True)
            elif filter_type == "Equalize":
                st.image(st.session_state.processed_img, caption="Processed Image (Equalization)", use_column_width=True)
            elif filter_type == "CLAHE":
                st.image(st.session_state.processed_img, caption="Processed Image (CLAHE)", use_column_width=True)
            elif filter_type == "Binary":
                st.image(st.session_state.processed_img, caption="Processed Image (Binary)", use_column_width=True)

        if st.button("Reset"):
            st.session_state.saved_img = None  # Reset saved image
            st.session_state.processed_img = None  # Reset processed image
            st.rerun()  # Rerun the app to reset the state

if __name__ == "__main__":
    main()
