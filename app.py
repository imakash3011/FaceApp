import streamlit as st
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os

def main():
    """Face App"""
    st.title("Face App")
    st.text("Build with streamlit and opencv")
    activities = ["Detection", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice == 'Detection':
        st.subheader(" Face Detection ")
    elif choice == 'About':
        st.subheader("About")


if __name__ == '__main__':
    main()

