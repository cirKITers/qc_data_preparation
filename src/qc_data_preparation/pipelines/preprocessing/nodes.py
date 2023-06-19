from typing import Dict, List, Tuple

import numpy as np


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
    return {
        "selected_x": x_values[class_mask],
        "selected_y": y_values[class_mask],
    }


def shuffle_data(
    x_values: np.ndarray, y_values: np.ndarray, seed: int
) -> Dict[str, np.ndarray]:
    assert len(x_values) == len(y_values), "Lenghts must be equal"
    rng = np.random.default_rng(seed=seed)
    permutation = rng.permutation(len(x_values))
    return {
        "shuffled_x": x_values[permutation],
        "shuffled_y": y_values[permutation],
    }


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
    return {
        "train_x": train_x,
        "train_y": train_y,
        "test_x": test_x,
        "test_y": test_y,
    }


def normalize_data(x_values: np.ndarray) -> np.ndarray:
    return np.divide(x_values, 255)


def add_channel_data(x_values: np.ndarray) -> np.ndarray:
    """
    Adding a (redundant) channel (dimension) to the data so that we can work on channels within the
    convolutional AE network

    Args:
        x_values (np.ndarray): data of shape [BxWxH]

    Returns:
        np.ndarray: data of shape [Bx1xWxH]
    """
    return x_values.reshape(x_values.shape[0], 1, x_values.shape[1], x_values.shape[2])


def sort_interleaved(x_values, y_values, classes) -> Dict[str, np.ndarray]:
    sort_order = []
    amount = len(y_values) // len(classes)
    for i in range(amount):
        for j in range(len(classes)):
            sort_order.append(j * amount + i)
    return {"sorted_x": x_values[sort_order], "sorted_y": y_values[sort_order]}
