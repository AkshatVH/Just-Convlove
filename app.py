import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

with st.sidebar:
    image = Image.open("C:\\Users\\Akshat\\Pictures\\robynne-hu-HOrhCnQsxnQ-unsplash.jpg")
    st.image(image, use_column_width= True)
    st.title(" JUST CONVOLVE ")
    choice = st.radio("Navigation", ["Upload", "ML"])
    st.info("This application allows you to run Convolution on your image with your filter!")

if os.path.exists("uploaded_file.png"):
    img = cv2.imread("uploaded_file.png")


if choice == "Upload":
    st.title("Input Image!!")
    file = st.file_uploader("Upload Your Image Here", type= 'png')
    #Uploading png Image
    if file is not None:
        file_bytes = np.asarray(bytearray(file.read()), dtype= np.uint8)
        opencv_img = cv2.imdecode(file_bytes, 1)
        cv2.imwrite("uploaded_file.png", opencv_img)

        st.image(file, caption = "Uploaded file", use_column_width=None)
        st.write("")
        st.write("Classifying...")

    # Uploading the Filter
    row = st.number_input("Enter the length of Filter", min_value= 1, max_value= 5)
    # arr = np.array(text)
    st.session_state.kernel = np.empty((row,row))

    for i in range(row):
        for j in range(row):
            st.session_state.kernel[i][j] = st.number_input("enter the values here ",min_value=-10, max_value=10, key = (i,j))

if choice == "ML":
    st.title("Performing Convolution")

    res = cv2.filter2D(img, -1, st.session_state.kernel)
    cv2.imwrite("after_convolution.png", res)
    st.image(res, caption = "After convolution", use_column_width=None)


    #Download the FILE
    with open("after_convolution.png", 'rb') as f:
        st.download_button("Download the Model", f, "after_convolution.png")

