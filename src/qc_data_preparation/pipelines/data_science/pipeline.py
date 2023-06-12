from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import (
    encode_data,
    generate_ssim_curve,
    generate_loss_curve,
    train_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                train_model,
                inputs=[
                    "shuffled_normalized_train_x",
                    "sorted_normalized_test_x",
                    "params:number_of_features",
                    "params:training.epochs",
                    "params:seed",
                    "params:fw_select",
                ],
                outputs=["autoencoder_model", "autoencoder_history"],
                name="train_model",
            ),
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
            "shuffled_normalized_train_x": "shuffled_normalized_train_x",
            "sorted_normalized_test_x": "sorted_normalized_test_x",
            "shuffled_train_y": "shuffled_train_y",
            "sorted_test_y": "sorted_test_y",
        },
        outputs={
            "encoded_test_data": "encoded_test_data",
            "encoded_train_data": "encoded_train_data",
        },
        namespace="data_science",
    )
