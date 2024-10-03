import numpy as np
import streamlit as st
import cv2
from PIL import Image

# Function to change the smoothing type
def choose_filter(filter_choice):
    # Change filters depending on choice input
    if filter_choice == "GaussianBlur":
        return "GaussianBlur"
    elif filter_choice == "MedianBlur":
        return "MedianBlur"
    elif filter_choice == "Normal":
        return "Normal"
    elif filter_choice == "Otsu":
        return "Otsu"

# Function to apply filters
def apply_filters(image, filter_type, adjustable_value):

    if filter_type == "Grayscale":              # Converts to single colour-channel image
        if len(image.shape) == 3:               # Only convert if it's a color image
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif filter_type == "GaussianBlur":         # Blurs based off a mathematical formula
        return cv2.GaussianBlur(image, (adjustable_value, adjustable_value), 0)
    elif filter_type == "MedianBlur":               # Blurs based off the median value of the kernel
        return cv2.medianBlur(image, adjustable_value)
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
        clahe = cv2.createCLAHE(adjustable_value)
        return clahe.apply(gray_image)
    elif filter_type == "Normal":
        if len(image.shape) == 3:               # Extra check to ensure Grayscale input image
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image
        _, binary_image = cv2.threshold(gray_image, adjustable_value, 255, cv2.THRESH_BINARY)
        return cv2.bitwise_not(binary_image)    # Invert the binary image
    elif filter_type == "Otsu":
        if len(image.shape) == 3:               # Extra check to ensure Grayscale input image
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image
        _, binary_image = cv2.threshold(gray_image, adjustable_value, 255, cv2.THRESH_OTSU)
        return cv2.bitwise_not(binary_image)    # Invert the binary image
    elif filter_type == "Erosion":
        if len(image.shape) == 3:               # Extra check to ensure Grayscale input image
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return cv2.erode(image, np.ones(adjustable_value, np.uint8), iterations=1)
    elif filter_type == "Dilation":
        if len(image.shape) == 3:               # Extra check
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return cv2.dilate(image, np.ones(adjustable_value, np.uint8), iterations=1)
    return image                                # Return the original image if filter type is None

def main():
    # Initialization
    # These variables need to be initialized in streamlit's unique session storage
    if 'saved_img' not in st.session_state:
        st.session_state.saved_img = None
    if 'processed_img' not in st.session_state:
        st.session_state.processed_img = None
    if 'history_images' not in st.session_state:
        st.session_state.history_images = []
    if 'history_text' not in st.session_state:
        st.session_state.history_text = []

    st.title("Image Processor")

    # Upload an image
    uploaded_file = st.file_uploader("Choose an Image to Process", type=["jpeg", "jpg", "png"])

    # Check if the uploaded file is None (indicating it has been removed)
    if uploaded_file is None and st.session_state.saved_img is not None:
        st.session_state.saved_img = None  # Reset saved image
        st.session_state.processed_img = None  # Reset processed image
        st.session_state.history_images = []  # Clear history
        st.session_state.history_text = []  # Clear history
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
            filter_choice = st.selectbox("Choose smoothing type", ["GaussianBlur", "MedianBlur"])
            chosen_filter = choose_filter(filter_choice)
            adjustable_value = st.slider("Choose Kernel Size (odd values only)", min_value=3, max_value=21, step=2, value=3)
            st.session_state.processed_img = apply_filters(st.session_state.saved_img, filter_choice, adjustable_value)
        elif filter_type == "Binary":
            filter_choice = st.selectbox("Choose binarization type", ["Normal", "Otsu"])
            chosen_filter = choose_filter(filter_choice)

            if filter_choice == "Normal":
                adjustable_value = st.slider("Choose Threshold Value: ", min_value=0, max_value=255, value=50)
                st.session_state.processed_img = apply_filters(st.session_state.saved_img, chosen_filter, adjustable_value)

            elif filter_choice == "Otsu":
                adjustable_value = 0
                st.session_state.processed_img = apply_filters(st.session_state.saved_img, chosen_filter,
                                                               adjustable_value)
        elif filter_type == "CLAHE":
            adjustable_value = st.slider("Enter the ClipLimit: ", min_value=1, max_value=30, value=5)
            st.session_state.processed_img = apply_filters(st.session_state.saved_img, filter_type, adjustable_value)
        elif filter_type != "None":
            adjustable_value = 0
            st.session_state.processed_img = apply_filters(st.session_state.saved_img, filter_type, adjustable_value)

        if st.button("Apply Filter"):
            # Update saved_img to the latest processed image only if filter was applied
            if filter_type != "None":
                st.session_state.saved_img = st.session_state.processed_img.copy()
                st.session_state.history_images.append(st.session_state.processed_img.copy())
                #st.session_state.history_text.append(f'{filter_type} - Kernel Size: {adjustable_value} / Threshold: {adjustable_value} / ClipLimit: {adjustable_value}')

            # Display the processed images
            if filter_type == "Grayscale":
                st.image(st.session_state.processed_img, caption="Processed Image (Grayscale)", use_column_width=True, channels="GRAY")
                st.session_state.history_text.append(
                    f'{filter_type}')

            elif filter_type == "Smoothing":
                st.image(st.session_state.processed_img, caption="Processed Image (Smoothing)", use_column_width=True)
                st.session_state.history_text.append(
                    f'{chosen_filter} - Kernel Size: {adjustable_value}')

            elif filter_type == "Equalize":
                st.image(st.session_state.processed_img, caption="Processed Image (Equalization)", use_column_width=True)
                st.session_state.history_text.append(
                    f'{filter_type}')

            elif filter_type == "CLAHE":
                st.image(st.session_state.processed_img, caption="Processed Image (CLAHE)", use_column_width=True)
                st.session_state.history_text.append(
                    f'{filter_type} - ClipLimit: {adjustable_value}')

            elif filter_type == "Erosion":
                st.image(st.session_state.processed_img, caption="Processed Image (Eroded)", use_column_width=True)
                st.session_state.history_text.append(
                    f'{filter_type} - Kernel Size: {adjustable_value}')

            elif filter_type == "Dilation":
                st.image(st.session_state.processed_img, caption="Processed Image (Dilation)", use_column_width=True)
                st.session_state.history_text.append(
                    f'{filter_type} - Kernel Size: {adjustable_value}')

            elif filter_type == "Binary":
                st.image(st.session_state.processed_img, caption="Processed Image (Binary)", use_column_width=True)
                st.session_state.history_text.append(
                    f'{chosen_filter} - Threshold: {adjustable_value}')

            if filter_type != None or filter_type == None: # Display history of applied filters
                st.text_area("History of Applied Filters", "\n".join(st.session_state.history_text), height=200)

        if st.button("Undo"):
            if st.session_state.history_images:
                st.session_state.history_images.pop()
                st.session_state.history_text.pop()

                if st.session_state.history_images:
                    st.session_state.processed_img = st.session_state.history_images[-1]
                    st.session_state.saved_img = st.session_state.processed_img.copy()
                    st.text_area("History of Applied Filters", "\n".join(st.session_state.history_text), height=200)
                else:
                    st.session_state.saved_img = None
                    st.session_state.processed_img = None

        if st.button("Reset"):
            st.session_state.saved_img = None  # Reset saved image
            st.session_state.processed_img = None  # Reset processed image
            st.session_state.history_images = []
            st.session_state.history_text = []
            st.rerun()  # Rerun the app to reset the state



if __name__ == "__main__":
    main()
