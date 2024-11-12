# Fashion Generation and Logo Printing

This project focuses on generating fashion images using a Generative Adversarial Network (GAN) model and allowing users to overlay logos on these generated images. The application enables the generation of fashion items like t-shirts, shirts, and tops, with a feature to upload and adjust logos on the selected clothing images.

## Project Overview

### Generated Clothing Images
The GAN model generates images of clothing items, specifically shirts, t-shirts, and tops, based on the latent vectors and category labels provided as input.

Here is an example of a generated clothing item image:

![Generated Cloth](https://github.com/bilalsxadad1231231/FASHION-GENERATION-AND-LOGO-PRINTING/blob/main/Pictures/generated%20cloth.png)

### Logo Fixing on Generated Images
The project also includes functionality for users to upload logos and overlay them on the generated clothing images. The user can adjust the position and size of the logo on the image, and save the combined image.

Here is an example of a clothing image with a logo fixed onto it:

![Logo Fixing](https://github.com/bilalsxadad1231231/FASHION-GENERATION-AND-LOGO-PRINTING/blob/main/Pictures/logo%20fixing.png)

## Features
- **Image Generation**: Generate fashion images from latent vectors using the GAN model.
- **Logo Upload**: Upload a logo image to overlay on the generated clothing images.
- **Logo Adjustment**: Adjust the size and position of the logo on the clothing image using sliders.
- **Image Saving**: Save the final image with the logo to the selected directory.

## Usage
1. Select a category (Shirt, T-shirt, Top).
2. Generate fashion images based on the selected category.
3. Upload a logo image (optional).
4. Adjust the logo's size and position on the generated image.
5. Save the final image with the logo overlay.

## Installation

To run this project, you need to install the following dependencies:

    ```bash
    pip install tensorflow numpy matplotlib streamlit Pillow

### Step 1: Download the Model Weights

Click on the link below to download the pre-trained model weights:

[Download Model Weights](https://drive.google.com/file/d/1BQHa-tUIipMHOwqPUQ5z6kzEs4VW9YcM/view?usp=sharing)

### Step 2: Place the Model Weights in the Correct Folder

Once you've downloaded the model weights file, extract it and place the weights file in the `generator` folder of the project directory.

### Step 3: Run the Application

After placing the model weights in the `generator` folder, you can run the application using Streamlit. Here's how to get started:

```bash
streamlit run app.py