import streamlit as st
import cv2
from PIL import Image
import numpy as np

def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Resize the image for consistency
    image = cv2.resize(image, (700, 300))

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to extract features
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    return image, gray, binary

# Function to detect bleed lines (these are found on the left of ₹500 and ₹2000 notes)
def detect_bleed_lines(binary_image):
    # Specify region of interest (ROI) for bleed lines
    roi = binary_image[30:100, 30:100]  # This may change based on specific notes
    lines = cv2.HoughLines(roi, 1, np.pi / 180, 200)

    if lines is not None:
        return True
    return False

# Function to detect Gandhi's watermark (template matching)
def detect_gandhi_watermark(gray_image, template_path='gandhi_template.jpg'):
    # Load the Gandhi watermark template
    template = cv2.imread(template_path, 0)
    w, h = template.shape[::-1]

    # Template matching to detect the watermark
    res = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.7
    loc = np.where(res >= threshold)

    if len(loc[0]) > 0:
        return True
    return False

# Function to detect security thread
def detect_security_thread(image):
    # Convert to HSV to filter thread color
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for detecting the thread's specific color (black/dark)
    lower_color = np.array([0, 0, 0])
    upper_color = np.array([180, 255, 30])

    mask = cv2.inRange(hsv, lower_color, upper_color)
    detected_thread = cv2.bitwise_and(image, image, mask=mask)

    thread_count = cv2.countNonZero(mask)
    if thread_count > 1000:  # Adjust based on specific note
        return True
    return False

# Function to detect and verify serial numbers using OCR (Tesseract)
def detect_serial_number(image):
    # Use Tesseract OCR to detect serial numbers
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(image, config=custom_config)

    # Process the text to check for serial number pattern
    serial_numbers = [line for line in text.split('\n') if line.isdigit()]
    return serial_numbers if serial_numbers else False

# Final currency detection function
def detect_fake_currency(image_path):
    image, gray_image, binary_image = preprocess_image(image_path)

    # Detect bleed lines
    bleed_lines_present = detect_bleed_lines(binary_image)
    
    # Detect Gandhi watermark
    gandhi_watermark_present = detect_gandhi_watermark(gray_image)
    
    # Detect security thread
    security_thread_present = detect_security_thread(image)
    
    # Detect serial number
    serial_numbers = detect_serial_number(image)

    return {
        "Bleed Lines Detected": bleed_lines_present,
        "Gandhi Watermark Detected": gandhi_watermark_present,
        "Security Thread Detected": security_thread_present,
        "Serial Numbers Detected": serial_numbers
    }
    
def is_currency_fake(detection_results):
    if (not detection_results["Bleed Lines Detected"] or
        not detection_results["Gandhi Watermark Detected"] or
        not detection_results["Security Thread Detected"]):
        return True
    return False
    
st.title("Indian Fake Currency Detection")
st.write("Upload an image of a ₹500 or ₹2000 currency note.")

uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Save the uploaded image
    image_path = "uploaded_image.jpg"
    image.save(image_path)

    # Run the detection algorithm
    detection_results = detect_fake_currency(image_path)

    # Display results
    st.write("Detection Results:")
    st.write(f"Bleed Lines Detected: {detection_results['Bleed Lines Detected']}")
    st.write(f"Gandhi Watermark Detected: {detection_results['Gandhi Watermark Detected']}")
    st.write(f"Security Thread Detected: {detection_results['Security Thread Detected']}")
    st.write(f"Serial Numbers: {detection_results['Serial Numbers Detected']}")

    # Fake currency verdict
    if is_currency_fake(detection_results):
        st.error("Fake Currency Detected!")
    else:
        st.success("Currency is Genuine.")
