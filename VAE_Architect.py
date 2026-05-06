# RUN THIS IN: tensorflow_env (conda)
# The "Archeologist" Script: Exploring the 16D Universe

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the "Brains" (The Decoder is the Generator)
# We only need the decoder to generate new things!
decoder = tf.keras.models.load_model("vae_ultra_decoder.keras", compile=False)

def explore_dimensions(n_dims=16, steps=10):
    digit_size = 28
    # We will create a big grid: 16 rows (one for each dimension)
    figure = np.zeros((digit_size * n_dims, digit_size * steps))
    
    print("Interrogating the 16 dimensions...")

    for dim in range(n_dims):
        # Linearly space the values for THIS specific dimension from -2 to 2
        grid_values = np.linspace(-2.0, 2.0, steps)
        
        for i, val in enumerate(grid_values):
            # Create a 16D vector of zeros
            z_sample = np.zeros((1, 16))
            # Only change the ONE dimension we are currently "touring"
            z_sample[0, dim] = val
            
            # Predict (Generate) the digit
            x_decoded = decoder.predict(z_sample, verbose=0)
            digit = x_decoded.reshape(digit_size, digit_size)
            
            # Place it in the master grid
            figure[dim * digit_size : (dim + 1) * digit_size,
                   i * digit_size : (i + 1) * digit_size] = digit

    # Plot the results
    plt.figure(figsize=(12, 20))
    plt.imshow(figure, cmap="Greys_r")
    plt.title("The 16 Dimensions of your VAE\n(Each row is one 'slider' from -2 to 2)")
    plt.ylabel("Dimension Index (0-15)")
    plt.xlabel("Value (from -2.0 to 2.0)")
    plt.yticks(np.arange(14, 16 * 28, 28), np.arange(16))
    plt.show()

explore_dimensions()