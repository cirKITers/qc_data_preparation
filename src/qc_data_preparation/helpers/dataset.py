import logging
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Tuple

import numpy as np
import tensorflow as tf
from kedro.io import AbstractDataSet
from keras.utils.data_utils import get_file


class TensorFlowDataset(AbstractDataSet):
    def __init__(self, filepath):
        self._filepath = PurePosixPath(filepath)

    def _load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: #train_x, train_y, text_x, text_y
        if not self._exists():
            logger = logging.getLogger(__name__)
            logger.info(f"Downloading dataset {self._filepath} from TensorFlow")
            origin_folder = (
                "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
            )
            get_file(
                self._filepath,
                origin=origin_folder + "mnist.npz",
                file_hash="731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1",  # noqa
            )
        return tf.keras.datasets.mnist.load_data(self._filepath)

    def _save(self, data: np.ndarray) -> None:
        return np.save(self._filepath, data)

    def _exists(self) -> bool:
        return Path(self._filepath.as_posix()).exists()

    def _describe(self) -> Dict[str, Any]:
        return dict(
            filepath=self._filepath
        )
