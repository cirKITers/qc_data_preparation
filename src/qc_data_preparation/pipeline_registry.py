"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from qc_data_preparation.pipelines import data_processing, data_science


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    data_processing_pipeline = data_processing.create_pipeline()
    data_science_pipeline = data_science.create_pipeline()
    return {
        "data_processing_pipeline": data_processing_pipeline,
        "data_science_pipeline": data_science_pipeline,
        "__default__": data_processing_pipeline + data_science_pipeline,
    }
