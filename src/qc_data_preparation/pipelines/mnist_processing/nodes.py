from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch as pt
import torchmetrics

from .autoencoder import TF_Autoencoder, PT_Autoencoder


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


def train_model(
    train_x, test_x, number_of_features, epochs, seed, fw_select
) -> Tuple[tf.keras.models.Model | pt.nn.Module, Dict]:

    ssim_sigma = 1.5
    # note that for torch, we cannot explicitly define the size of the gaussian filter
    # for tf however, there is no different option than using gaussian
    ssim_filter_size = 11 
    ssim_k1 = 0.01
    ssim_k2 = 0.03

    if fw_select == "TensorFlow":
        tf.random.set_seed(seed)

        # defining custom ssim to set the parameters accordingly
        # note that we are not using multiscale_ssim although tf should make use of batches
        # using multiscale_ssim however, results in error
        def ssim(pred, target):
            return tf.image.ssim(pred, target, max_val=1.0, filter_size=ssim_filter_size, filter_sigma=ssim_sigma, k1=ssim_k1, k2=ssim_k2)

        autoencoder = TF_Autoencoder(number_of_features)
        autoencoder.compile(
            optimizer="adam", loss=tf.keras.losses.MeanSquaredError(), metrics=[ssim]
        )
        history = autoencoder.fit(
            train_x, train_x, epochs=epochs, shuffle=True, validation_data=(test_x, test_x)
        )

        history = history.history

    elif fw_select == "PyTorch":
        pt.manual_seed(seed)
        autoencoder = PT_Autoencoder(number_of_features)

        # Adam optimizer as in TF implementation
        optimizer = pt.optim.Adam(autoencoder.parameters())
        # MSE loss as in TF implementation
        loss_fct = pt.nn.MSELoss()
        # use the SSIM for calculating what is the "accuracy" in TensorFlow
        ssim = torchmetrics.StructuralSimilarityIndexMeasure(sigma=ssim_sigma, kernel_size=ssim_filter_size, k1=ssim_k1, k2=ssim_k2, data_range=1.0)

        # adding the channel dimension
        train_x = pt.Tensor(add_channel_data(train_x))
        test_x = pt.Tensor(add_channel_data(test_x))

        # generate two datasets with x=y
        train_dataset = pt.utils.data.TensorDataset(train_x, train_x)
        test_dataset = pt.utils.data.TensorDataset(test_x, test_x)
        
        # from those datasets, generate data loaders that take care of the shuffling and splitting in batches
        train_dataloader = pt.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=32)
        test_dataloader = pt.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=32)

        # mimic the structure from tensorflow
        history={
            "loss":[],
            "accuracy":[],
            "val_loss":[],
            "val_accuracy":[]
        }

        for epoch in range(epochs):
            # put the model in training mode
            autoencoder.train()

            epoch_loss = []
            epoch_accuracy = []
            for data, target in train_dataloader:
                output = autoencoder(data)  # forward
                loss = loss_fct(output, target) # loss calculation
                optimizer.zero_grad() # zero gradients before optimization
                loss.backward() # calculate loss
                optimizer.step() # backward

                epoch_loss.append(loss.item())
                epoch_accuracy.append(ssim(output, target).item())

            # get the mean loss and accuracy from the batch data
            history['loss'].append(np.mean(epoch_loss))
            history['accuracy'].append(np.mean(epoch_accuracy))

            # put the model in validation/test mode
            autoencoder.eval()

            # do not care about gradients now
            with pt.no_grad():
                epoch_loss = []
                epoch_accuracy = []
                for data, target in test_dataloader:
                    output = autoencoder(data) # forward
                    loss = loss_fct(output, target) # loss calculation

                    epoch_loss.append(loss.item()) # detach loss
                    epoch_accuracy.append(ssim(output, target).item()) # calculate similarity

            # get the mean loss and accuracy from the batch data
            history['val_loss'].append(np.mean(epoch_loss))
            history['val_accuracy'].append(np.mean(epoch_accuracy))
    else:
        raise RuntimeError(f"Unknown framework selected: {fw_select}. Framework must be one of [PyTorch, TensorFlow]")


    return autoencoder, history


def encode_data(
    model: tf.keras.models.Model | pt.nn.Module,
    values_x: np.ndarray,
    values_y: np.ndarray,
    classes: List,
) -> Dict[str, Any]:

    # check based on the type if we are using TF or PT framework
    if type(model) == PT_Autoencoder:
        with pt.no_grad(): # have to use no_grad as the tensor requires grad otherwise
            features = model.encoder(pt.Tensor(add_channel_data(values_x))).numpy()
    elif type(model) == TF_Autoencoder:
        features = model.encoder(values_x).numpy()
    else:
        raise RuntimeError(f"Unknown model type: {type(model)}. Model must be one of [PT_Autoencoder, TF_Autoencoder]")

    return {
        "labels": values_y,
        "features": features,
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


def generate_ssim_curve(history: Dict):
    loss_train = history["ssim"]
    loss_val = history["val_ssim"]
    epochs = range(1, len(list(history.values())[0]) + 1)
    plt.plot(epochs, loss_train, "g", label="Training SSIM")
    plt.plot(epochs, loss_val, "b", label="Validation SSIM")
    plt.title("Training and Validation SSIM")
    plt.xlabel("Epochs")
    plt.ylabel("SSIM")
    plt.legend()
    return plt
