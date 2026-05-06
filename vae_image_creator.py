# RUN THIS IN: tensorflow_env (conda)
# "ULTRA" VAE: Increased Complexity, Higher Fidelity

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Step 2: Load MNIST ---
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# --- Step 3: Set "Ultra" Hyperparameters ---
latent_dim = 16  # Sharper images, but harder to visualize
batch_size = 128
epochs = 50      # More training time for better results

# --- Step 4 & 5: Model Definitions ---
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Encoder (Doubled Filters: 64 and 128)
encoder_inputs = layers.Input(shape=(28, 28, 1))
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(32, activation="relu")(x) # Increased dense layer size
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = models.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

# Decoder (Doubled Filters: 128 and 64)
latent_inputs = layers.Input(shape=(latent_dim,))
y = layers.Dense(7 * 7 * 128, activation="relu")(latent_inputs)
y = layers.Reshape((7, 7, 128))(y)
y = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(y)
y = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(y)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(y)
decoder = models.Model(latent_inputs, decoder_outputs, name="decoder")

# --- Step 6: Connect VAE ---
class VAE(models.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(tf.keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2))
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"loss": total_loss, "reconstruction": reconstruction_loss, "kl": kl_loss}

vae = VAE(encoder, decoder)
vae.compile(optimizer=tf.keras.optimizers.Adam())

# --- Step 8: Train (Watch the fan!) ---
print("Starting heavy training...")
vae.fit(x_train, epochs=epochs, batch_size=batch_size)

# --- Step 9: Plot a "Slice" of the Latent Space ---
def plot_latent_space(vae, n=15, digit_size=28):
    figure = np.zeros((digit_size * n, digit_size * n))
    grid_x = np.linspace(-1.0, 1.0, n)
    grid_y = np.linspace(-1.0, 1.0, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.zeros((1, latent_dim)) # Create a row of 16 zeros
            z_sample[0, 0] = xi                 # Set the 1st "slider"
            z_sample[0, 1] = yi                 # Set the 2nd "slider"
            
            x_decoded = vae.decoder.predict(z_sample, verbose=0)
            digit = x_decoded.reshape(digit_size, digit_size)
            figure[i * digit_size : (i + 1) * digit_size,
                   j * digit_size : (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.title(f"16D Latent Space (Slice of first 2 dims)")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()

plot_latent_space(vae)

# --- SAVE THE MODELS ---
print("\nSaving high-fidelity models...")
encoder.save("vae_ultra_encoder.keras")
decoder.save("vae_ultra_decoder.keras")
print("Success! Models saved.")