"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from qc_data_preparation.pipelines import (
    preprocessing,
    training,
    postprocessing,
)


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    preprocessing_pipeline = preprocessing.create_pipeline()
    training_pipeline = training.create_pipeline()
    postprocessing_pipeline = postprocessing.create_pipeline()
    return {
        "preprocessing": preprocessing_pipeline,
        "tf_training": training_pipeline["tf_training_pipeline"],
        "pt_training": training_pipeline["pt_training_pipeline"],
        "postprocessing": postprocessing_pipeline,
        "__default__": preprocessing_pipeline
        + training_pipeline["pt_training_pipeline"]
        + postprocessing_pipeline,
    }
