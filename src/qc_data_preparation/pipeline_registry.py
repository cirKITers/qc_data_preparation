"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from qc_data_preparation.pipelines import mnist_processing


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    mnist_processing_pipeline = mnist_processing.create_pipeline()
    return {
        "mnist_processing": mnist_processing_pipeline,
        "__default__": mnist_processing_pipeline,
    }
