import streamlit as st
import easyocr
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'])  # You can add more languages if needed

# Streamlit UI Setup
st.title("Handwritten Text Recognition with OCR")
st.write("Upload an image with handwritten text, and the app will extract the text using OCR.")

# File uploader to upload an image
uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert the image to OpenCV format (numpy array)
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    # Perform OCR
    results = reader.readtext(image_cv)

    # Draw Bounding Boxes and Display Results
    for (bbox, text, prob) in results:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))

        # Draw bounding box and text
        cv2.rectangle(image_cv, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(image_cv, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 0, 0), 2)

    # Convert the image back to RGB format for displaying
    image_cv_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

    # Display the processed image with bounding boxes
    st.image(image_cv_rgb, caption="Processed Image with OCR Results",
             use_column_width=True)

    # Print the extracted text
    st.subheader("Extracted Text:")
    for bbox, text, prob in results:
        st.write(f"- {text} (Confidence: {prob:.2f})")
