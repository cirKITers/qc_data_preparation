import sys
import logging
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Tuple

import numpy as np
import tensorflow as tf
from kedro.io import AbstractDataSet
from kedro.extras.datasets.plotly import JSONDataSet
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


# MacOS Patch for loading of directories
# This is mostly a copy of TensorFlowModelDataSet._load but
# explicitly treats the to-be-copied directory as such.
if sys.platform == "darwin":
    import tempfile
    from pathlib import PurePath
    from kedro.io.core import get_filepath_str
    from kedro_datasets.tensorflow import TensorFlowModelDataSet
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
