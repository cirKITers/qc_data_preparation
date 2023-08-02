from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
import torch as pt
import torchmetrics

from .autoencoder import TF_Autoencoder
from .autoencoder import PT_Autoencoder_Exp as PT_Autoencoder


def add_channel_data(x_values: np.ndarray) -> np.ndarray:
    """
    Adding a (redundant) channel (dimension) to the data so that we can work on channels
    within the convolutional AE network

    Args:
        x_values (np.ndarray): data of shape [BxWxH]

    Returns:
        np.ndarray: data of shape [Bx1xWxH]
    """
    return x_values.reshape(x_values.shape[0], 1, x_values.shape[1], x_values.shape[2])


def train_tf_model(
    train_x,
    test_x,
    number_of_features,
    epochs,
    seed,
    batch_size,
    ssim_filter_size,
    ssim_sigma,
    ssim_k1,
    ssim_k2,
) -> Tuple[tf.keras.models.Model, Dict]:
    tf.random.set_seed(seed)

    # defining custom ssim to set the parameters accordingly
    # we need that for saving the model (load via {'ssim':ssim})
    @tf.keras.utils.register_keras_serializable()
    def ssim(pred, target):
        # add a BW channel (N, H, W) -> (N, H, W, 1)
        # TODO: move that in preprocessing (requires adjustment of model to return channel)
        pred = tf.expand_dims(pred, axis=3) if len(pred.shape) == 3 else pred 
        target = tf.expand_dims(target, axis=3) if len(target.shape) == 3 else target 

        return tf.image.ssim(
            pred,
            target,
            max_val=1.0,
            filter_size=ssim_filter_size,
            filter_sigma=ssim_sigma,
            k1=ssim_k1,
            k2=ssim_k2,
        )

    autoencoder = TF_Autoencoder(number_of_features)
    autoencoder.compile(
        optimizer="adam",
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[ssim],
    )
    history = autoencoder.fit(
        train_x,
        train_x,
        epochs=epochs,
        shuffle=True,
        validation_data=(test_x, test_x),
        batch_size=batch_size,
        verbose=0,
    )

    history = history.history

    return autoencoder, history


def train_pt_model(
    train_x,
    test_x,
    number_of_features,
    epochs,
    seed,
    batch_size,
    ssim_filter_size,
    ssim_sigma,
    ssim_k1,
    ssim_k2,
) -> Tuple[pt.nn.Module, Dict]:
    pt.manual_seed(seed)

    autoencoder = PT_Autoencoder(number_of_features)

    # Adam optimizer as in TF implementation
    optimizer = pt.optim.Adam(autoencoder.parameters())
    # MSE loss as in TF implementation
    loss_fct = pt.nn.MSELoss()
    # use the SSIM for calculating what is the "accuracy" in TensorFlow
    ssim = torchmetrics.StructuralSimilarityIndexMeasure(
        sigma=ssim_sigma,
        kernel_size=ssim_filter_size,
        k1=ssim_k1,
        k2=ssim_k2,
        data_range=1.0,
    )

    # adding the channel dimension
    train_x = pt.Tensor(add_channel_data(train_x))
    test_x = pt.Tensor(add_channel_data(test_x))

    # generate two datasets with x=y
    train_dataset = pt.utils.data.TensorDataset(train_x, train_x)
    test_dataset = pt.utils.data.TensorDataset(test_x, test_x)

    # from those datasets, generate data loaders that take care of the
    # shuffling and splitting in batches
    train_dataloader = pt.utils.data.DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size
    )
    test_dataloader = pt.utils.data.DataLoader(
        test_dataset, shuffle=True, batch_size=batch_size
    )

    # mimic the structure from tensorflow
    history = {"loss": [], "ssim": [], "val_loss": [], "val_ssim": []}

    for epoch in range(epochs):
        # put the model in training mode
        autoencoder.train()

        epoch_loss = []
        epoch_ssim = []
        for data, target in train_dataloader:
            output = autoencoder(data)  # forward
            loss = loss_fct(output, target)  # loss calculation
            optimizer.zero_grad()  # zero gradients before optimization
            loss.backward()  # calculate loss
            optimizer.step()  # backward

            epoch_loss.append(loss.item())
            epoch_ssim.append(ssim(output, target).item())

        # get the mean loss and ssim from the batch data
        history["loss"].append(np.mean(epoch_loss))
        history["ssim"].append(np.mean(epoch_ssim))

        # put the model in validation/test mode
        autoencoder.eval()

        # do not care about gradients now
        with pt.no_grad():
            epoch_loss = []
            epoch_ssim = []
            for data, target in test_dataloader:
                output = autoencoder(data)  # forward
                loss = loss_fct(output, target)  # loss calculation

                epoch_loss.append(loss.item())  # detach loss
                epoch_ssim.append(ssim(output, target).item())  # calculate similarity

        # get the mean loss and ssim from the batch data
        history["val_loss"].append(np.mean(epoch_loss))
        history["val_ssim"].append(np.mean(epoch_ssim))

    return autoencoder, history
