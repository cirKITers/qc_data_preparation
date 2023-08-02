from typing import Any, Dict, List

import numpy as np
import tensorflow as tf
import torch as pt
import plotly.graph_objects as go

from ..training.nodes import add_channel_data


def encode_data(
    model: "tf.keras.models.Model | pt.nn.Module",
    values_x: np.ndarray,
    values_y: np.ndarray,
    classes: List,
) -> Dict[str, Any]:
    # check based on the type if we are using TF or PT framework
    if issubclass(type(model), pt.nn.Module):
        with pt.no_grad():  # have to use no_grad as the tensor requires grad otherwise
            features = model.encoder(pt.Tensor(add_channel_data(values_x))).numpy()
    elif issubclass(type(model), tf.keras.Model):
        features = model.encoder(values_x).numpy()
    else:
        raise TypeError(
            f"Unknown model type: {type(model)}. Model must be one of [PT_Autoencoder,"
            " TF_Autoencoder]"
        )

    return {
        "labels": values_y,
        "features": features,
        "classes": classes,
    }


def generate_loss_curve(history: Dict):
    loss_train = history["loss"]
    loss_val = history["val_loss"]
    epochs = range(1, len(list(history.values())[0]) + 1)

    plt = go.Figure(
        [
            go.Scatter(
                x=list(epochs),
                y=loss_train,
                mode="lines+markers",
                name="Training Loss",
            ),
            go.Scatter(
                x=list(epochs),
                y=loss_val,
                mode="lines+markers",
                name="Validation Loss",
            ),
        ]
    )
    plt.update_layout(
        title="Training and Validation Loss",
        xaxis_title="Epochs",
        yaxis_title="Loss",
    )

    return plt


def generate_ssim_curve(history: Dict):
    ssim_train = history["ssim"]
    ssim_val = history["val_ssim"]
    epochs = range(1, len(list(history.values())[0]) + 1)

    plt = go.Figure(
        [
            go.Scatter(
                x=list(epochs),
                y=ssim_train,
                mode="lines+markers",
                name="Training SSIM",
            ),
            go.Scatter(
                x=list(epochs),
                y=ssim_val,
                mode="lines+markers",
                name="Validation SSIM",
            ),
        ]
    )
    plt.update_layout(
        title="Training and Validation SSIM",
        xaxis_title="Epochs",
        yaxis_title="SSIM",
    )

    return plt
