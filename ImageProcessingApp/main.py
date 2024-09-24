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
    # Initialization
    if 'key' not in st.session_state:
        st.session_state['key'] = 'value'

    if 'count' not in st.session_state:
        st.session_state.count = 0

    if 'saved_img' not in st.session_state:
        st.session_state.saved_img = None

    def set_count(i):
        st.session_state.count = i

    st.title("Image Processor")

    uploaded_file = st.file_uploader("Choose an Image to Process", type=["jpeg", "jpg", "png"], on_change=set_count, args=[0]) #accepts JPEG, PNG, and JPG image types for processing

    if uploaded_file is not None: #Checks if file is uploaded
        st.write("File Uploaded")
        img = Image.open(uploaded_file)  # Uses PIL library to open image
        img_array = np.array(img)  # Image converted to NumPy array, so it can be processed by OpenCV, array represents pixel values (RBG format)

        if st.session_state.count == 0:
            st.image(img, caption="Uploaded Image", use_column_width=True)

            # Display the current count
            st.write("Current count:", st.session_state.count)

        filter_type = st.selectbox("Choose a Filter", ["None", "Grayscale", "Smoothing"], key="filter_choice", on_change=set_count, args=[1])

        if st.session_state.count == 1:

            processed_img = apply_filters(img_array, filter_type)

            saved_img = processed_img

            if filter_type == "Grayscale":
                st.image(processed_img, caption="Processed Image", use_column_width=True, channels="GRAY") #If Greyscale filter chosen

            elif filter_type == "Smoothing":
                st.image(processed_img, caption="Processed Image", use_column_width=True) #If Smoothing or none - both are in RBG

            # elif filter_type == "None":
            #     st.session_state.count = 0

            if st.button("Apply New Filter"):
                st.write("Button clicked!")
                st.session_state.count += 1
                st.rerun()

            if st.button("Revert", on_click=set_count, args=[0]):
                st.write("Reverted")
                st.session_state.count = 0
                st.rerun()

            st.slider("Choose a value")

            # Display the current count
            st.write("Current count:", st.session_state.count)

if __name__ == "__main__":
    main()
