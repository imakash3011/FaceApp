import streamlit as st
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os


@st.cache
def load_image(img):
    im = Image.open(img)
    return im


def main():
    """Face App"""
    st.title("Face App")
    st.text("Build with streamlit and opencv")
    activities = ["Detection", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice == 'Detection':
        st.subheader(" Face Detection ")

        image_file = st.file_uploader("Please Upload an Image")
        if image_file is not None:
            our_image = Image.open(image_file)
            st.text("Original Image")
            # st.write(type(our_image))
            st.image(our_image)

        enhance_type = st.sidebar.radio("Enhance Type", ["Original", "Gray-Scale", "Contrast","Brightness", "Blurring"])
        if enhance_type == 'Gray-Scale':
            new_img = np.array(our_image.convert('RGB'))
            # img = cv2.cvtColor(new_img, 1)
            gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
            # st.write(new_img)
            st.image(gray)
        
        



    elif choice == 'About':
        st.subheader("About")


if __name__ == '__main__':
    main()

