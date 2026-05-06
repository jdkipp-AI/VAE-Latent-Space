# RUN THIS IN: tensorflow_env (conda)
# The "Evolution" Script: Morphing Real Handwritten Digits

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1. We need the Sampling class defined so Keras can load the model
class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# 2. Load the "Brains" (Custom objects required for the Sampling layer)
encoder = tf.keras.models.load_model("vae_ultra_encoder.keras", custom_objects={'Sampling': Sampling})
decoder = tf.keras.models.load_model("vae_ultra_decoder.keras")

# 3. Load MNIST test data to find "Real" targets
(_, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_test = x_test.astype('float32') / 255.
x_test = np.expand_dims(x_test, -1)

def evolve_digits(start_idx=0, end_idx=1, steps=10):
    # Get the two real images
    img_start = x_test[start_idx:start_idx+1]
    img_end = x_test[end_idx:end_idx+1]

    # Encode them into 16D space
    z_start_mean, _, _ = encoder.predict(img_start, verbose=0)
    z_end_mean, _, _ = encoder.predict(img_end, verbose=0)

    # Create the "Evolution" path (Linear Interpolation)
    # We move from z_start to z_end in 'steps'
    alphas = np.linspace(0, 1, steps)
    
    plt.figure(figsize=(20, 4))
    
    for i, alpha in enumerate(alphas):
        # The Math: (1-alpha)*Start + alpha*End
        z_interpolated = (1 - alpha) * z_start_mean + alpha * z_end_mean
        
        # Decode the "In-Between" thought
        reconstructed_img = decoder.predict(z_interpolated, verbose=0)
        
        # Plotting
        ax = plt.subplot(1, steps, i + 1)
        plt.imshow(reconstructed_img.reshape(28, 28), cmap="Greys_r")
        plt.axis("off")
        if i == 0: plt.title("Start (Real)")
        if i == steps - 1: plt.title("End (Real)")

    plt.suptitle(f"The 16D Evolution: Index {start_idx} to {end_idx}")
    plt.show()

# Run it! Change these indices to pick different numbers from the test set
# (Try 0 and 1, or 4 and 15, etc.)
evolve_digits(start_idx=2, end_idx=110, steps=12)