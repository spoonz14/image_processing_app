import numpy
import streamlit
import cv2


def apply_filters(self):
    return True



def main():

    import streamlit as st
    st.title("Image Processor")

    uploaded_file = st.file_uploader("Choose an Image to Process", type = "JPEG")

    if uploaded_file is not None:
        st.write("File Uploaded")

main()
