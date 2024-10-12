import cv2
import numpy as np
import streamlit as st
from PIL import Image

def load_image(image_file):
    img = Image.open(image_file)
    return img

def feature_matching(image1, image2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Initiate ORB detector
    orb = cv2.ORB_create()

    # Find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw first 50 matches
    result = cv2.drawMatches(image1, kp1, image2, kp2, matches[:50], None, flags=2)

    return result, len(matches)

def process_images(real_image_path, test_image_path):
    # Read images
    real_img = cv2.imread(real_image_path)
    test_img = cv2.imread(test_image_path)

    # Resize images to a fixed size
    real_img = cv2.resize(real_img, (500, 300))
    test_img = cv2.resize(test_img, (500, 300))

    # Feature matching
    result_img, match_count = feature_matching(real_img, test_img)

    return result_img, match_count

def detect_fake_currency(match_count, threshold=100):
    if match_count > threshold:
        return "The currency is REAL!"
    else:
        return "The currency is FAKE!"

# Streamlit app code
def main():
    st.title("Fake Currency Detection System")
    
    st.write("Upload an image of the real currency and the test currency.")

    # Uploading real and test currency images
    real_image_file = st.file_uploader("Upload Real Currency", type=["jpg", "png", "jpeg"])
    test_image_file = st.file_uploader("Upload Test Currency", type=["jpg", "png", "jpeg"])

    if real_image_file is not None and test_image_file is not None:
        real_img = load_image(real_image_file)
        test_img = load_image(test_image_file)

        # Convert PIL images to OpenCV format
        real_img = np.array(real_img)
        test_img = np.array(test_img)

        # Run currency detection
        result_img, match_count = feature_matching(real_img, test_img)

        # Display the number of matches
        st.write(f"Feature matches: {match_count}")

        # Determine if the currency is fake or real
        detection_result = detect_fake_currency(match_count)
        st.subheader(detection_result)

        # Display result image with matches
        st.image(result_img, channels="BGR", use_column_width=True)

if __name__ == "__main__":
    main()
