import tensorflow as tf


class Autoencoder(tf.keras.models.Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512),
                tf.keras.layers.Dense(256),
                tf.keras.layers.Dense(128),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(64),
                tf.keras.layers.Dense(32),
                tf.keras.layers.Lambda(lambda x: tf.math.sin(x)),
                tf.keras.layers.Dense(16),
                tf.keras.layers.Dense(latent_dim),
                tf.keras.layers.Lambda(lambda x: tf.math.sin(x)),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(16),
                tf.keras.layers.Dense(32),
                tf.keras.layers.Lambda(lambda x: tf.math.sin(x)),
                tf.keras.layers.Dense(64),
                tf.keras.layers.Dense(128),
                tf.keras.layers.Dense(256),
                tf.keras.layers.Dense(512),
                tf.keras.layers.Dense(784, activation="sigmoid"),
                tf.keras.layers.Reshape((28, 28)),
            ]
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
