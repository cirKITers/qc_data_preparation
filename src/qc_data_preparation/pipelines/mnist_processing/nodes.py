from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from .autoencoder import Autoencoder


def concat_data(
    data: Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
) -> Dict[str, np.ndarray]:
    (train_x, train_y), (test_x, test_y) = data
    all_x = np.concatenate((train_x, test_x))
    all_y = np.concatenate((train_y, test_y))
    return {"all_x": all_x, "all_y": all_y}


def subset_data(
    x_values: np.ndarray, y_values: np.ndarray, classes: List
) -> Dict[str, np.ndarray]:
    class_mask = np.isin(y_values, classes)
    return {"selected_x": x_values[class_mask], "selected_y": y_values[class_mask]}


def shuffle_data(
    x_values: np.ndarray, y_values: np.ndarray, seed: int
) -> Dict[str, np.ndarray]:
    assert len(x_values) == len(y_values), "Lenghts must be equal"
    rng = np.random.default_rng(seed=seed)
    permutation = rng.permutation(len(x_values))
    return {"shuffled_x": x_values[permutation], "shuffled_y": y_values[permutation]}


def split_data(
    x_values: np.ndarray,
    y_values: np.ndarray,
    train_ipc: int,
    test_ipc: int,
) -> Dict[str, np.ndarray]:
    train_x, train_y, test_x, test_y = None, None, None, None
    for cls in np.unique(y_values):
        cls_mask = np.isin(y_values, cls)
        cls_x_values = x_values[cls_mask]
        cls_y_values = y_values[cls_mask]
        assert len(cls_x_values) >= train_ipc + test_ipc
        if train_x is not None:
            train_x = np.concatenate((train_x, cls_x_values[:train_ipc]))
            # TODO: check errors
            train_y = np.concatenate((train_y, cls_y_values[:train_ipc]))
            test_x = np.concatenate((test_x, cls_x_values[-test_ipc:]))
            test_y = np.concatenate((test_y, cls_y_values[-test_ipc:]))
        else:
            train_x = cls_x_values[:train_ipc]
            train_y = cls_y_values[:train_ipc]
            test_x = cls_x_values[-test_ipc:]
            test_y = cls_y_values[-test_ipc:]
    return {"train_x": train_x, "train_y": train_y, "test_x": test_x, "test_y": test_y}


def normalize_data(x_values: np.ndarray) -> np.ndarray:
    return np.divide(x_values, 255)


def sort_interleaved(x_values, y_values, classes) -> Dict[str, np.ndarray]:
    sort_order = []
    amount = len(y_values) // len(classes)
    for i in range(amount):
        for j in range(len(classes)):
            sort_order.append(j * amount + i)
    return {"sorted_x": x_values[sort_order], "sorted_y": y_values[sort_order]}


def train_model(
    train_x, test_x, number_of_features, epochs, seed
) -> Tuple[tf.keras.models.Model, Dict]:
    tf.random.set_seed(seed)
    autoencoder = Autoencoder(number_of_features)
    autoencoder.compile(
        optimizer="adam", loss=tf.keras.losses.MeanSquaredError(), metrics=["accuracy"]
    )
    history = autoencoder.fit(
        train_x, train_x, epochs=epochs, shuffle=True, validation_data=(test_x, test_x)
    )
    return autoencoder, history.history


def encode_data(
    model: tf.keras.models.Model,
    values_x: np.ndarray,
    values_y: np.ndarray,
    classes: List,
) -> Dict[str, Any]:
    return {
        "labels": values_y,
        "features": model.encoder(values_x).numpy(),
        "classes": classes,
    }


def generate_loss_curve(history: Dict):
    loss_train = history["loss"]
    loss_val = history["val_loss"]
    epochs = range(1, len(list(history.values())[0]) + 1)
    plt.plot(epochs, loss_train, "g", label="Training loss")
    plt.plot(epochs, loss_val, "b", label="Validation loss")
    plt.title("Training and Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    return plt


def generate_accuracy_curve(history: Dict):
    loss_train = history["accuracy"]
    loss_val = history["val_accuracy"]
    epochs = range(1, len(list(history.values())[0]) + 1)
    plt.plot(epochs, loss_train, "g", label="Training accuracy")
    plt.plot(epochs, loss_val, "b", label="Validation accuracy")
    plt.title("Training and Validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    return plt
