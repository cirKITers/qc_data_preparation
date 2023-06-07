import tensorflow as tf
import torch as pt

class TF_Autoencoder(tf.keras.models.Model):
    def __init__(self, latent_dim):
        super(TF_Autoencoder, self).__init__()
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

class PT_Autoencoder(pt.nn.Module):
    """
    Autoencoder based on the approach in
    https://covalent.readthedocs.io/en/latest/tutorials/0_ClassicalMachineLearning/autoencoders/source.html
    """

    def __init__(self, latent_space_dim):
        super(PT_Autoencoder, self).__init__()
        self.encoder = pt.nn.Sequential(
            pt.nn.Conv2d(
                1, 8, 3, stride=1, padding=1
            ),  # input size = 1x28x28 -> hidden size = 8x28x28
            pt.nn.ReLU(True),
            pt.nn.Dropout(p=0.5),
            pt.nn.Conv2d(
                8, 16, 3, stride=2, padding=1
            ),  # input size = 8x28x28 -> hidden size = 16x14x14
            pt.nn.ReLU(True),
            pt.nn.Dropout(p=0.3),
            pt.nn.Conv2d(
                16, 32, 3, stride=2, padding=1
            ),  # hidden size = 16x14x14 -> hidden size = 32x7x7
            pt.nn.ReLU(True),
            pt.nn.Conv2d(32, 64, 7),  # hidden size = 32x7x7 -> hidden size = 64x1x1
            pt.nn.Tanh(),
            pt.nn.Flatten(),
            pt.nn.Linear(64, 16),  # hidden size = 64x1x1 -> hidden size = latent_space_dim x1x1
            pt.nn.Tanh(),
            pt.nn.Linear(16, latent_space_dim),  # hidden size = 64x1x1 -> hidden size = latent_space_dim x1x1
            pt.nn.Tanh()
        )

        self.decoder = pt.nn.Sequential(
            pt.nn.Linear(latent_space_dim, 16),  # hidden size = latent_space_dim x1x1 -> hidden size = 64x1x1
            pt.nn.ReLU(True),
            pt.nn.Linear(16, 64),  # hidden size = 64x1x1 -> hidden size = latent_space_dim x1x1
            pt.nn.ReLU(True),
            pt.nn.Unflatten(1, (64, 1, 1)),
            pt.nn.ConvTranspose2d(64, 32, 7),  # input size = latent_space_dim x1x1 -> hidden size = 32x7x7
            pt.nn.ReLU(True),
            pt.nn.ConvTranspose2d(
                32, 16, 3, stride=2, padding=1, output_padding=1
            ),  # hidden size = 32x7x7 -> hidden size = 16x14x14
            pt.nn.ReLU(True),
            pt.nn.Dropout(p=0.3),
            pt.nn.ConvTranspose2d(
                16, 8, 3, stride=2, padding=1, output_padding=1
            ),  # hidden size = 16x14x14 -> hidden size = 8x28x28
            pt.nn.ReLU(True),
            pt.nn.Dropout(p=0.5),
            pt.nn.ConvTranspose2d(
                8, 1, 3, stride=1, padding=1, output_padding=0
            ),  # hidden size = 8x28x28 -> hidden size = 1x28x28
            pt.nn.Sigmoid(),  # output with pixels in [0,1]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class PT_Autoencoder_Exp(pt.nn.Module):
    """
    Autoencoder based on the approach in
    https://covalent.readthedocs.io/en/latest/tutorials/0_ClassicalMachineLearning/autoencoders/source.html
    """

    def __init__(self, latent_space_dim):
        super(PT_Autoencoder_Exp, self).__init__()
        self.encoder = pt.nn.Sequential(
            pt.nn.Conv2d(
                1, 8, 3, stride=1, padding=1
            ),  # input size = 1x28x28 -> hidden size = 8x28x28
            pt.nn.ReLU(True),
            pt.nn.Dropout(p=0.5),
            pt.nn.Conv2d(
                8, 16, 3, stride=2, padding=1
            ),  # input size = 8x28x28 -> hidden size = 16x14x14
            pt.nn.ReLU(True),
            pt.nn.Dropout(p=0.3),
            pt.nn.Conv2d(
                16, 32, 3, stride=2, padding=1
            ),  # hidden size = 16x14x14 -> hidden size = 32x7x7
            pt.nn.ReLU(True),
            pt.nn.Conv2d(32, 64, 7),  # hidden size = 32x7x7 -> hidden size = 64x1x1
            pt.nn.Tanh(),
            pt.nn.Flatten(),  # hidden size = 64x1x1 -> hidden size = 64
            pt.nn.Linear(64, 16),  # hidden size = 64 -> hidden size = 16
            pt.nn.Tanh(),
            pt.nn.Linear(16, latent_space_dim),  # hidden size = 16 -> hidden size = latent_space_dim
            pt.nn.Tanh()
        )

        self.pre_decoder_a = pt.nn.Sequential(
            pt.nn.Linear(latent_space_dim, 16),  # hidden size = latent_space_dim -> hidden size = 16
            pt.nn.ReLU(True),
            pt.nn.Linear(16, 32),  # hidden size = 16 -> hidden size = 64
            pt.nn.ReLU(True),
            pt.nn.Unflatten(1, (32, 1, 1)),
        )
        self.pre_decoder_b = pt.nn.Sequential(
            pt.nn.Linear(latent_space_dim, 32),  # hidden size = latent_space_dim -> hidden size = 16
            pt.nn.ReLU(True),
            pt.nn.Unflatten(1, (32, 1, 1)),
        )
        self.decoder = pt.nn.Sequential(
            pt.nn.ConvTranspose2d(64, 32, 7),  # input size = latent_space_dim x1x1 -> hidden size = 32x7x7
            pt.nn.ReLU(True),
            pt.nn.ConvTranspose2d(
                32, 16, 3, stride=2, padding=1, output_padding=1
            ),  # hidden size = 32x7x7 -> hidden size = 16x14x14
            pt.nn.ReLU(True),
            pt.nn.Dropout(p=0.3),
            pt.nn.ConvTranspose2d(
                16, 8, 3, stride=2, padding=1, output_padding=1
            ),  # hidden size = 16x14x14 -> hidden size = 8x28x28
            pt.nn.ReLU(True),
            pt.nn.Dropout(p=0.5),
            pt.nn.ConvTranspose2d(
                8, 1, 3, stride=1, padding=1, output_padding=0
            ),  # hidden size = 8x28x28 -> hidden size = 1x28x28
            pt.nn.Sigmoid(),  # output with pixels in [0,1]
        )

    def forward(self, x):
        encoded = self.encoder(x)

        pre_decoded_a = self.pre_decoder_a(encoded)
        pre_decoded_b = self.pre_decoder_b(encoded)

        decoded = self.decoder(pt.cat([pre_decoded_a, pre_decoded_b], dim=1))
        return decoded