import logging
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Tuple

import numpy as np
from qc_data_preparation.pipelines.training.autoencoder import (
    PT_Autoencoder,
    TF_Autoencoder,
)
import tensorflow as tf
import torch
from kedro.io import AbstractDataSet
from kedro.extras.datasets.plotly import JSONDataSet
from kedro.extras.datasets.tensorflow import TensorFlowModelDataset
from keras.utils.data_utils import get_file

import plotly.graph_objects as go


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
        load_args: Dict[str, Any] = None,
        save_args: Dict[str, Any] = None,
    ) -> None:
        self._filepath = PurePosixPath(filepath)
        self._load_args = load_args
        self._save_args = save_args

    def _load(self) -> torch.nn.Module:
        state_dict = torch.load(self._filepath)
        model = PT_Autoencoder(**self._load_args)
        model.load_state_dict(state_dict)
        return model

    def _save(self, model: torch.nn.Module) -> None:
        torch.save(model.state_dict(), self._filepath, **self._save_args)

    def _exists(self) -> bool:
        return Path(self._filepath.as_posix()).exists()

    def _describe(self) -> Dict[str, Any]:
        return dict(
            filepath=self._filepath,
            model=PT_Autoencoder,
            load_args=self._load_args,
            save_args=self._save_args,
        )


class TFPTModelDataset(AbstractDataSet):
    def __init__(
        self,
        filepath: str,
        model: "torch.nn.Module | tf.keras.Model",
        load_args: Dict[str, Any] = None,
        save_args: Dict[str, Any] = None,
    ) -> None:
        logger = logging.getLogger(__name__)
        if issubclass(type(model), torch.nn.Module) or model == "PT_Autoencoder":
            # provide PTModelDataset
            logger.info(f"Handling model {model} as pytorch model")
            self._model_io = PTModelDataset(
                filepath=filepath, load_args=load_args, save_args=save_args
            )
        elif issubclass(type(model), tf.keras.Model) or model == "TF_Autoencoder":
            # provide TFModelDataset
            logger.info(f"Handling model {model} as tensorflow model")
            self._model_io = TensorFlowModelDataset(
                filepath=filepath, load_args=load_args, save_args=save_args
            )
        else:
            logger.error(f"Type of model {model} is not provided")
            raise NotImplementedError

    def _load(self) -> "torch.nn.Module | tf.keras.Model":
        return self._model_io._load()

    def _save(self, model: "torch.nn.Module | tf.keras.Model") -> None:
        self._model_io._save(model)

    def _exists(self) -> bool:
        return self._model_io._exists()

    def _describe(self):
        return self._model_io._describe()
