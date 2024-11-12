import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def generate_images(model, test_input, save_dir='generated_images', category = 1):
    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    output = None
    for label in range(3):
        labels = tf.ones(10) * category
        predictions = model([test_input, labels], training=False)
        if output is None:
            output = predictions
        else:
            output = np.concatenate((output, predictions))

    nrow = 3
    ncol = 10
    
    k = 0
    for i in range(nrow):
        for j in range(ncol):
            pred = (output[k, :, :, :] + 1) * 127.5  # Rescale to [0, 255]
            pred = np.array(pred)
            
            # Convert the prediction to uint8 format
            pred_uint8 = pred.astype(np.uint8)
            
            # Define the filename and save the image
            filename = os.path.join(save_dir, f'image_{k}.png')
            plt.imsave(filename, pred_uint8)
            
            k += 1

    print(f'Images saved to directory: {save_dir}')

# Example usage:
# generate_images(your_model, your_test_input, save_dir='/kaggle/working/generated_images')
