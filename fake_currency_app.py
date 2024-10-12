import streamlit as st
import cv2
from PIL import Image
import numpy as np

# Load reference templates for comparison
gandhi_template = cv2.imread('gandhi_watermark_template.jpg', 0)
ashoka_pillar_template = cv2.imread('ashoka_pillar_template.jpg', 0)

# Functions for detecting security features (as defined earlier)
# Load the reference images (templates for Gandhi's watermark, security thread, etc.)

# Preprocess the input currency image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
    return img, gray, thresh

# Feature 1: Detect Gandhi's Watermark using Template Matching
def detect_gandhi_watermark(gray_image):
    result = cv2.matchTemplate(gray_image, gandhi_template, cv2.TM_CCOEFF_NORMED)
    (_, max_val, _, max_loc) = cv2.minMaxLoc(result)
    
    # Set a threshold for matching (0.7 is usually a good threshold)
    threshold = 0.7
    if max_val > threshold:
        # print("Gandhi Watermark detected")
        return True
    else:
        # print("Gandhi Watermark not detected")
        return False

# Feature 2: Detect Ashoka Pillar using Template Matching
def detect_ashoka_pillar(gray_image):
    result = cv2.matchTemplate(gray_image, ashoka_pillar_template, cv2.TM_CCOEFF_NORMED)
    (_, max_val, _, max_loc) = cv2.minMaxLoc(result)
    
    threshold = 0.7
    if max_val > threshold:
        # print("Ashoka Pillar detected")
        return True
    else:
        # print("Ashoka Pillar not detected")
        return False

# Feature 3: Detect Bleed Lines using Edge Detection
def detect_bleed_lines(gray_image):
    edges = cv2.Canny(gray_image, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Set a threshold for number of bleed lines
    min_bleed_lines = 7
    detected_bleed_lines = 0
    for contour in contours:
        if cv2.arcLength(contour, True) > 100:  # Adjust based on the size of bleed lines
            detected_bleed_lines += 1
    
    if detected_bleed_lines >= min_bleed_lines:
        # print("Bleed Lines detected")
        return True
    else:
        # print("Bleed Lines not detected")
        return False

# Detect Security Features
def detect_security_features(image_path):
    img, gray, thresh = preprocess_image(image_path)
    
    # Run all the detection algorithms
    gandhi_watermark = detect_gandhi_watermark(gray)
    ashoka_pillar = detect_ashoka_pillar(gray)
    bleed_lines = detect_bleed_lines(gray)

    return gandhi_watermark, ashoka_pillar, bleed_lines

st.title("Fake Currency Detection for Indian Notes (500, 2000)")

st.write("Upload an image of the currency note below:")

# File uploader for currency image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # Preprocess the uploaded image and detect features
    _, gray, _ = preprocess_image(image_np)
    
    st.image(image, caption='Uploaded Currency Note.', use_column_width=True)

    # Detect security features
    gandhi_watermark, ashoka_pillar, bleed_lines = detect_security_features(image_np)

    # Display results
    st.write("Results:")
    st.write(f"Gandhi Watermark: {'Detected' if gandhi_watermark else 'Not Detected'}")
    st.write(f"Ashoka Pillar: {'Detected' if ashoka_pillar else 'Not Detected'}")
    st.write(f"Bleed Lines: {'Detected' if bleed_lines else 'Not Detected'}")

    if gandhi_watermark and ashoka_pillar and bleed_lines:
        st.success("This currency note seems genuine.")
    else:
        st.error("This currency note might be fake.")
