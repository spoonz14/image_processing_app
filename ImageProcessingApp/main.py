import numpy as np
import streamlit as st
import cv2
from PIL import Image

# Function to change the smoothing type
def choose_smoothing(smoothing_choice):
    # Change filters depending on choice input
    if smoothing_choice == "GaussianBlur":
        return "GaussianBlur"
    elif smoothing_choice == "Median":
        return "Median"

# Function to apply filters
def apply_filters(image, filter_type, kernel, threshold, clip_limit):

    if filter_type == "Grayscale":              # Converts to single colour-channel image
        if len(image.shape) == 3:               # Only convert if it's a color image
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif filter_type == "GaussianBlur":         # Blurs based off a mathematical formula
        return cv2.GaussianBlur(image, (kernel, kernel), 0)
    elif filter_type == "Median":               # Blurs based off the median value of the kernel
        return cv2.medianBlur(image, kernel)
    elif filter_type == "Equalize":             # Stretches the average value across the image
        if len(image.shape) == 3:               # Extra check to ensure single channel/Grayscale input image
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            return cv2.equalizeHist(gray_image)
        return cv2.equalizeHist(image)
    elif filter_type == "CLAHE":
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image
        clahe = cv2.createCLAHE(clip_limit)
        return clahe.apply(gray_image)
    elif filter_type == "Binary":
        if len(image.shape) == 3:               # Extra check to ensure Grayscale input image
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image
        _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return cv2.bitwise_not(binary_image)    # Invert the binary image
    elif filter_type == "Erosion":
        if len(image.shape) == 3:               # Extra check to ensure Grayscale input image
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return cv2.erode(image, np.ones(kernel, np.uint8), iterations=1)
    elif filter_type == "Dilation":
        if len(image.shape) == 3:               # Extra check
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return cv2.dilate(image, np.ones(kernel, np.uint8), iterations=1)
    return image                                # Return the original image if filter type is None

def main():
    # Initialization
    # These variables need to be initialized in streamlit's unique session storage
    if 'saved_img' not in st.session_state:
        st.session_state.saved_img = None
    if 'processed_img' not in st.session_state:
        st.session_state.processed_img = None

    st.title("Image Processor")

    # Upload an image
    uploaded_file = st.file_uploader("Choose an Image to Process", type=["jpeg", "jpg", "png"])

    # Check if the uploaded file is None (indicating it has been removed)
    if uploaded_file is None and st.session_state.saved_img is not None:
        st.session_state.saved_img = None  # Reset saved image
        st.session_state.processed_img = None  # Reset processed image
        st.rerun()  # Rerun the app to reset the state

    if uploaded_file is not None:
        st.write("File Uploaded")
        img = Image.open(uploaded_file)
        img_array = np.array(img)           # Save image as a numpy array for OpenCV


        # Display the uploaded image only once
        if st.session_state.saved_img is None:
            st.image(img, caption="Uploaded Image", use_column_width=True)
            st.session_state.saved_img = img_array
            st.session_state.processed_img = img_array.copy()  # Initialize processed image

        elif st.session_state.saved_img is not None:
            # Display the original uploaded image
            st.image(st.session_state.saved_img, caption="Original Image", use_column_width=True)



        filter_type = st.selectbox("Choose a Filter", ["None", "Grayscale", "Smoothing", "Equalize", "CLAHE", "Binary", "Dilation", "Erosion" ],
                                   key="filter_choice")


        if filter_type == "Smoothing":
            smoothing_choice = st.selectbox("Choose smoothing type", ["GaussianBlur", "Median"])
            chosen_smoothing = choose_smoothing(smoothing_choice)
            kernel = st.slider("Choose Kernel Size (odd values only)", min_value=3, max_value=21, step=2, value=3)
            threshold = 0  # Smoothing doesnt use this
            clip_limit = 0 # Or this
            st.session_state.processed_img = apply_filters(st.session_state.saved_img, chosen_smoothing, kernel, threshold, clip_limit)
        elif filter_type == "Binary":
            kernel = 0 # Not used by Binary filter
            threshold = st.slider("Choose Threshold Value: ", min_value=0, max_value=255, value=50)
            clip_limit = 0 # Not used by Binary FIlter
            st.session_state.processed_img = apply_filters(st.session_state.saved_img, filter_type, kernel, threshold, clip_limit)
        elif filter_type == "CLAHE":
            kernel = 0 # Not used by CLAHE
            threshold = 0 # Not used by CLAHE
            clip_limit = st.slider("Enter the ClipLimit: ", min_value=1, max_value=30, value=5)
        elif filter_type != "None":
            threshold = 0
            kernel = 0
            clip_limit = 0
            st.session_state.processed_img = apply_filters(st.session_state.saved_img, filter_type, kernel, threshold, clip_limit)

        if st.button("Apply Filter"):
            # Update saved_img to the latest processed image only if filter was applied
            if filter_type != "None":
                st.session_state.saved_img = st.session_state.processed_img.copy()

            # Display the processed images
            if filter_type == "Grayscale":
                st.image(st.session_state.processed_img, caption="Processed Image (Grayscale)", use_column_width=True, channels="GRAY")

            elif filter_type == "Smoothing":
                st.image(st.session_state.processed_img, caption="Processed Image (Smoothing)", use_column_width=True)

            elif filter_type == "Equalize":
                st.image(st.session_state.processed_img, caption="Processed Image (Equalization)", use_column_width=True)

            elif filter_type == "CLAHE":
                st.image(st.session_state.processed_img, caption="Processed Image (CLAHE)", use_column_width=True)

            elif filter_type == "Erosion":
                st.image(st.session_state.processed_img, caption="Processed Image (Eroded)", use_column_width=True)

            elif filter_type == "Dilation":
                st.image(st.session_state.processed_img, caption="Processed Image (Dilation)", use_column_width=True)

            elif filter_type == "Binary":
                st.image(st.session_state.processed_img, caption="Processed Image (Binary)", use_column_width=True)

        if st.button("Reset"):
            st.session_state.saved_img = None  # Reset saved image
            st.session_state.processed_img = None  # Reset processed image
            st.rerun()  # Rerun the app to reset the state

if __name__ == "__main__":
    main()
