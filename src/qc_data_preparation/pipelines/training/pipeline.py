from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import (
    train_tf_model,
    train_pt_model
)


def create_pipeline(**kwargs) -> Pipeline:
    tf_training_pipeline = pipeline(
        [
            node(
                train_tf_model,
                inputs=[
                    "shuffled_normalized_train_x",
                    "sorted_normalized_test_x",
                    "params:number_of_features",
                    "params:epochs",
                    "params:seed",
                    "params:batch_size",
                    "params:ssim_filter_size",
                    "params:ssim_sigma",
                    "params:ssim_k1",
                    "params:ssim_k2",
                ],
                outputs=["autoencoder_model", "autoencoder_history"],
                name="train_model",
            ),
        ],
        inputs={
            "shuffled_normalized_train_x": "shuffled_normalized_train_x",
            "sorted_normalized_test_x": "sorted_normalized_test_x",
        },
        outputs={
            "autoencoder_model": "autoencoder_model",
            "autoencoder_history": "autoencoder_history",
        },
        namespace="training",
    )

    pt_training_pipeline = pipeline(
        [
            node(
                train_pt_model,
                inputs=[
                    "shuffled_normalized_train_x",
                    "sorted_normalized_test_x",
                    "params:number_of_features",
                    "params:epochs",
                    "params:seed",
                    "params:batch_size",
                    "params:ssim_filter_size",
                    "params:ssim_sigma",
                    "params:ssim_k1",
                    "params:ssim_k2",
                ],
                outputs=["autoencoder_model", "autoencoder_history"],
                name="train_model",
            ),
        ],
        inputs={
            "shuffled_normalized_train_x": "shuffled_normalized_train_x",
            "sorted_normalized_test_x": "sorted_normalized_test_x",
        },
        outputs={
            "autoencoder_model": "autoencoder_model",
            "autoencoder_history": "autoencoder_history",
        },
        namespace="training",
    )

    return {
        "tf_training_pipeline":tf_training_pipeline,
        "pt_training_pipeline":pt_training_pipeline,
    }
