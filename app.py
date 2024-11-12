import streamlit as st
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from model import generator
from utils import generate_images
import shutil
import numpy as np
from io import BytesIO

# Directories
IMAGE_DIR = "generated_images"
SELECTED_DIR = "selected"

# Ensure directories exist
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(SELECTED_DIR, exist_ok=True)

# Function to clear existing images from the directory
def clear_images(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
# Sidebar for logo upload
st.sidebar.header("Logo Upload")
logo_file = st.sidebar.file_uploader("Upload a logo image", type=["png", "jpg", "jpeg"])

# Streamlit application
st.title("Fashion Image Generator")
label_dict = {0: 'shirt', 1: 'tshirt', 2: 'top'}
category_dict = {'Shirt': 0, 'T-shirt': 1, 'Top': 2}

# Tabs
tab1, tab2 = st.tabs(["Generate & Select Images", "View & Edit Selected Image"])

# Tab 1: Generate & Select Images
with tab1:
    # Dropdown menu to select category
    category = st.selectbox("Select Category", ["Shirt", "T-shirt", "Top"])

    # Button to generate images
    if st.button("Generate"):
        # Clear existing images
        clear_images(IMAGE_DIR)
        clear_images(SELECTED_DIR)
        # Generate new images based on the selected category
        with st.spinner("Generating images..."):
            try:
                latent_dim = 100
                noise = tf.random.normal([10, latent_dim])
                generate_images(generator, noise, category=category_dict[category])

                st.success("Images generated successfully!")
            except Exception as e:
                st.error(f"Error generating images: {e}")

    # Display images in a grid
    st.header("Generated Images")
    selected_image_path = None

    for i in range(0, 10, 4):
        cols = st.columns(4)
        for j in range(4):
            image_path = os.path.join(IMAGE_DIR, f"image_{i + j}.png")
            if os.path.exists(image_path):
                # Display the image
                cols[j].image(image_path, caption=f"{category} {i + j + 1}")

                # If the button is clicked, move the image to the selected directory
                if cols[j].button(f"Select Image {i + j + 1}", key=image_path):
                    selected_image_path = image_path
                    selected_image_name = f"selected_image_{i + j}.png"
                    selected_image_dest = os.path.join(SELECTED_DIR, selected_image_name)

                    # Move the selected image to the selected directory
                    shutil.copyfile(image_path, selected_image_dest)
                    st.success(f"Image {i + j + 1} selected and moved to '{SELECTED_DIR}' directory.")

# Tab 2: View & Edit Selected Image
# Tab 2: View & Edit Selected Image
# Tab 2: View & Edit Selected Image
# Tab 2: View & Edit Selected Image
with tab2:
    # Load the image from the selected directory
    image_files = [f for f in os.listdir(SELECTED_DIR) if f.startswith('selected_image_') and f.endswith('.png')]

    if image_files:
        selected_image_path = os.path.join(SELECTED_DIR, image_files[0])  # Load the first image in the directory

        if os.path.exists(selected_image_path):
            selected_image = Image.open(selected_image_path)
            
            # Resize the image to a fixed size for display
            display_size = (800, 800)  # Adjust the size as needed
            selected_image = selected_image.resize(display_size)
            
            # Initialize variables for logo position
            if 'logo_pos_x' not in st.session_state:
                st.session_state.logo_pos_x = (selected_image.width - 128) // 2
            if 'logo_pos_y' not in st.session_state:
                st.session_state.logo_pos_y = (selected_image.height - 128) // 2

            # Overlay the logo if uploaded
            if logo_file:
                logo_image = Image.open(logo_file)

                # Slider for resizing the logo
                logo_size = st.slider("Select Logo Size", min_value=50, max_value=300, value=128)

                # Resize the logo based on slider value
                logo_image = logo_image.resize((logo_size, logo_size))

                # Create a copy of the selected image to overlay the logo
                combined_image = selected_image.copy()

                # Update logo position based on session state
                combined_image.paste(logo_image, (st.session_state.logo_pos_x, st.session_state.logo_pos_y), logo_image)
                st.image(combined_image, caption="Image with Logo", use_column_width=False)

                # Save the combined image to the selected directory
                if st.button("Save Image"):
                    combined_image_path = os.path.join(SELECTED_DIR, f"combined_{os.path.basename(selected_image_path)}")
                    combined_image.save(combined_image_path)
                    st.success(f"Combined image saved as '{combined_image_path}'")

            # Navigation buttons
            st.subheader("Edit Logo Position")
            col1, col2, col3, col4 = st.columns(4)
            speed = st.slider("Logo Movement Speed", min_value=10, max_value=50, value=10)

            # Button click handlers
            move_up = col1.button("Up")
            move_down = col2.button("Down")
            move_left = col3.button("Left")
            move_right = col4.button("Right")

            if move_up:
                st.session_state.logo_pos_y = max(0, st.session_state.logo_pos_y - speed)
            if move_down:
                st.session_state.logo_pos_y = min(selected_image.height - logo_size, st.session_state.logo_pos_y + speed)
            if move_left:
                st.session_state.logo_pos_x = max(0, st.session_state.logo_pos_x - speed)
            if move_right:
                st.session_state.logo_pos_x = min(selected_image.width - logo_size, st.session_state.logo_pos_x + speed)

    else:
        st.write("No images found in the selected directory.")
