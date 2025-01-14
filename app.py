# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---





# +
import cv2
import os
import numpy as np
import streamlit as st
import zipfile

# Function to process the image and extract red blood cells
def process_image(uploaded_file):
    # Load the image
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to segment the red blood cells
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find the contours of the segmented red blood cells
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a directory to save the individual red blood cell images
    if not os.path.exists('cells'):
        os.makedirs('cells')

    # Save each red blood cell as a separate image file
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        cell_image = image[y:y+h, x:x+w]
        cv2.imwrite(f'cells/cell_{i+1}.png', cell_image)

    # Delete images less than 1KB (noise)
    for filename in os.listdir('cells'):
        file_path = os.path.join('cells', filename)
        if os.path.isfile(file_path) and os.path.getsize(file_path) < 10024:
            os.remove(file_path)

    return contours

# Function to remove background and save images
def remove_background_and_save():
    input_dir = './cells'
    output_dir = './RBC'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = os.listdir(input_dir)
    for idx, image_file in enumerate(image_files, start=1):
        image_path = os.path.join(input_dir, image_file)
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i, contour in enumerate(contours, start=1):
            x, y, w, h = cv2.boundingRect(contour)
            cell_image = image[y:y+h, x:x+w]
            cell_mask = np.zeros_like(image)
            cv2.drawContours(cell_mask, [contour], 0, (255, 255, 255), thickness=cv2.FILLED)
            cell_mask = cv2.resize(cell_mask, (cell_image.shape[1], cell_image.shape[0]))
            cell_image = cv2.bitwise_and(cell_image, cell_mask)
            cell_filename = f'imagemdx_{idx}_{i:02d}.png'
            cell_path = os.path.join(output_dir, cell_filename)
            cv2.imwrite(cell_path, cell_image)

    # Delete images less than 1KB (noise)
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        if os.path.isfile(file_path) and os.path.getsize(file_path) < 10024:
            os.remove(file_path)

# Function to create a ZIP file of the segmented images
def create_zip_of_images(output_dir):
    zip_filename = "segmented_cells.zip"
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for root, _, files in os.walk(output_dir):
            for file in files:
                zipf.write(os.path.join(root, file), file)
    return zip_filename

# Streamlit app layout
st.title("Red Blood Cell Segmentation")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    st.image(image, channels="BGR", caption='Uploaded Image', use_container_width=True)

    # Reset the uploaded file pointer for processing
    uploaded_file.seek(0)

    # Process the image and count contours
    contours = process_image(uploaded_file)
    st.write(f"Found {len(contours)} red blood cell(s).")
    
    # Remove background and save images
    remove_background_and_save()
    
    # Create ZIP file of the segmented images
    if os.path.exists('./RBC'):
        zip_file_path = create_zip_of_images('./RBC')
        
        # Provide a download button for the ZIP file
        with open(zip_file_path, "rb") as f:
            st.download_button(
                label="Download All Segmented Images",
                data=f,
                file_name=zip_file_path,
                mime="application/zip"
            )

# -


