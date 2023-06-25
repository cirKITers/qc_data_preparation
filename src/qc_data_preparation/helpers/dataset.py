import os
import sys
import logging
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Tuple

import fsspec
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import get_file
import torch
from kedro.io import AbstractDataSet
from kedro.io.core import get_protocol_and_path
from kedro_datasets.tensorflow import TensorFlowModelDataSet
from kedro_datasets.plotly import JSONDataSet
import plotly.graph_objects as go

from ..pipelines.training.autoencoder import PT_Autoencoder_Exp as PT_Autoencoder


class PlotlyDataSet(JSONDataSet):
    def _save(self, data: go.Figure) -> None:
        """
        Saves the provided pandas dataframe into both, a plotly json file
        (for further processing) and a html file (for easy access via browser).
        This method should be adapted if anything changes in the implementation
        of the parent's class method.
        """
        data.write_html(self._filepath.with_suffix(".html").as_posix(), full_html=True)

        return super()._save(data)


# MacOS Patch for loading of directories
# This is mostly a copy of TensorFlowModelDataSet._load but
# explicitly treats the to-be-copied directory as such.
if sys.platform == "darwin":
    import tempfile
    from pathlib import PurePath
    from kedro.io.core import get_filepath_str
    from kedro_datasets.tensorflow.tensorflow_model_dataset import TEMPORARY_H5_FILE

    def load_folder(self) -> tf.keras.Model:
        load_path = get_filepath_str(self._get_load_path(), self._protocol)

        with tempfile.TemporaryDirectory(prefix=self._tmp_prefix) as path:
            if self._is_h5:
                path = str(PurePath(path) / TEMPORARY_H5_FILE)
                self._fs.copy(load_path, path)
            else:
                # MacOS Patch: add / to force copying the folder content only
                self._fs.get(load_path + "/", path, recursive=True)

            # Pass the local temporary directory/file path to keras.load_model
            device_name = self._load_args.pop("tf_device", None)
            if device_name:
                with tf.device(device_name):
                    model = tf.keras.models.load_model(path, **self._load_args)
            else:
                model = tf.keras.models.load_model(path, **self._load_args)
            return model

    TensorFlowModelDataSet._load = load_folder


class TensorFlowDataset(AbstractDataSet):
    def __init__(self, filepath):
        self._filepath = PurePosixPath(filepath)

    def _load(
        self,
    ) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:  # train_x, train_y, text_x, text_y
        if not self._exists():
            logger = logging.getLogger(__name__)
            logger.info(f"Downloading dataset {self._filepath} from TensorFlow")
            origin_folder = (
                "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
            )
            get_file(
                self._filepath,
                origin=origin_folder + "mnist.npz",
                file_hash=(  # noqa
                    "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"
                ),
            )
        return tf.keras.datasets.mnist.load_data(self._filepath)

    def _save(self, data: np.ndarray) -> None:
        return np.save(self._filepath, data)

    def _exists(self) -> bool:
        return Path(self._filepath.as_posix()).exists()

    def _describe(self) -> Dict[str, Any]:
        return dict(filepath=self._filepath)


class PTModelDataset(AbstractDataSet):
    def __init__(
        self,
        filepath: str,
        load_args: "Dict[str, Any] | None" = None,
        save_args: "Dict[str, Any] | None" = None,
    ) -> None:
        self._tmp_prefix = "kedro_pytorch_tmp"  # temp prefix pattern
        self._filepath = filepath
        self._load_args = load_args
        self._save_args = save_args
        _fs_args = {}
        self._protocol, _ = get_protocol_and_path(filepath)
        if self._protocol == "file":
            _fs_args.setdefault("auto_mkdir", True)
        self._fs = fsspec.filesystem(self._protocol, **_fs_args)

    def _load(self) -> torch.nn.Module:
        state_dict = torch.load(self._filepath)
        model = PT_Autoencoder(**self._load_args)
        model.load_state_dict(state_dict)
        return model

    def _save(self, model: torch.nn.Module) -> None:
        save_path = self._filepath
        with tempfile.TemporaryDirectory(prefix=self._tmp_prefix) as tmp_dir:
            tmp_path = tmp_dir + "/" + os.path.basename(save_path)
            torch.save(model.state_dict(), tmp_path, **self._save_args)
            if self._fs.exists(save_path):
                self._fs.rm(save_path, recursive=True)
            self._fs.put(tmp_path, save_path, recursive=False)

    def _exists(self) -> bool:
        return self._fs.exists(self._filepath)

    def _describe(self) -> Dict[str, Any]:
        return dict(
            filepath=self._filepath,
            model=PT_Autoencoder,
            load_args=self._load_args,
            save_args=self._save_args,
        )


class TFPTModelDataset(AbstractDataSet):
    """
    Dataset that supports saving and loading of tensorflow and pytorch models
    based on :py:class:`~.PT_Autoencoder` and :py:class:`~.TF_Autoencoder`.
    """

    def __init__(
        self,
        filepath: str,
        load_args: "Dict[str, Any] | None" = None,
        save_args: "Dict[str, Any] | None" = None,
    ) -> None:
        self._save_args = {} if not save_args else save_args
        self._load_args = {} if not load_args else load_args
        self._filepath = filepath
        self._model_io = None

    def _load(self) -> "torch.nn.Module | tf.keras.Model":
        """
        This method tries to load one of the supported models,
        :py:class::`~.PT_Autoencoder`, or :py:class::`~.TF_Autoencoder`.
        """
        if self._model_io is not None:
            return self._model_io._load()
        for model_init in (self._init_tf_io, self._init_pt_io):
            try:
                model_init()
                return self._model_io._load()
            except IsADirectoryError:
                pass
            raise TypeError(f"Type of model in {self._filepath} is not supported")

    def _init_pt_io(self):
        self._model_io = PTModelDataset(
            filepath=self._filepath,
            load_args=self._load_args,
            save_args=self._save_args,
        )

    def _init_tf_io(self):
        """
        This method ensures that the ssim metric can be loaded by putting the
        specific method into `custom_objects` for `load_args`. However, it
        currently ignores the set parameters from data catalog. So it cannot
        be perfectly restored at the moment.
        """

        @tf.keras.utils.register_keras_serializable()
        def ssim(pred, target):
            return tf.image.ssim(
                pred,
                target,
                max_val=1.0,
                filter_size=11,
                filter_sigma=1.5,
                k1=0.01,
                k2=0.03,
            )

        load_args = self._load_args.copy()
        del load_args["latent_dim"]  # needs to be deleted for proper loading
        load_args["custom_objects"] = {"ssim": ssim}
        self._model_io = TensorFlowModelDataSet(
            filepath=self._filepath,
            load_args=load_args,
            save_args=self._save_args,
        )

    def _save(self, model: "torch.nn.Module | tf.keras.Model") -> None:
        logger = logging.getLogger(__name__)
        if issubclass(type(model), torch.nn.Module):
            # provide PTModelDataset
            logger.info(f"Handling model {model} as pytorch model")
            self._init_pt_io()
        elif issubclass(type(model), tf.keras.Model):
            # provide TFModelDataset
            logger.info(f"Handling model {model} as tensorflow model")
            self._init_tf_io()
        else:
            raise TypeError(f"Type of model {model} is not provided")
        self._model_io._save(model)

    def _exists(self) -> bool:
        return self._model_io._exists()

    def _describe(self) -> Dict[str, Any]:
        return dict(
            filepath=self._filepath,
            model=self._model_io,
            load_args=self._load_args,
            save_args=self._save_args,
        )
