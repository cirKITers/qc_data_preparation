from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import (
    encode_data,
    generate_ssim_curve,
    generate_loss_curve,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                encode_data,
                inputs=[
                    "autoencoder_model",
                    "shuffled_normalized_train_x",
                    "shuffled_train_y",
                    "params:classes",
                ],
                outputs="encoded_train_data",
                name="encode_train_data",
            ),
            node(
                encode_data,
                inputs=[
                    "autoencoder_model",
                    "sorted_normalized_test_x",
                    "sorted_test_y",
                    "params:classes",
                ],
                outputs="encoded_test_data",
                name="encode_test_data",
            ),
            node(
                generate_loss_curve,
                inputs="autoencoder_history",
                outputs="loss_curve",
                name="generate_loss_curve",
            ),
            node(
                generate_ssim_curve,
                inputs="autoencoder_history",
                outputs="accuracy_curve",
                name="generate_accuracy_curve",
            )
        ],
        inputs={
            "autoencoder_model": "autoencoder_model",
            "autoencoder_history": "autoencoder_history",
            "shuffled_normalized_train_x": "shuffled_normalized_train_x",
            "sorted_normalized_test_x": "sorted_normalized_test_x",
            "shuffled_train_y": "shuffled_train_y",
            "sorted_test_y": "sorted_test_y",
        },
        outputs={
            "loss_curve": "loss_curve",
            "accuracy_curve": "accuracy_curve",
            "encoded_test_data": "encoded_test_data",
            "encoded_train_data": "encoded_train_data",
        },
        namespace="postprocessing",
    )
