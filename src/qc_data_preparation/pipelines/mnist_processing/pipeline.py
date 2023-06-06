from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import (
    concat_data,
    encode_data,
    generate_ssim_curve,
    generate_loss_curve,
    normalize_data,
    shuffle_data,
    sort_interleaved,
    split_data,
    subset_data,
    train_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                concat_data,
                inputs="mnist_data",
                outputs={"all_x": "all_x", "all_y": "all_y"},
                name="concat_data",
            ),
            node(
                subset_data,
                inputs=["all_x", "all_y", "params:classes"],
                outputs={"selected_x": "selected_x", "selected_y": "selected_y"},
                name="subset_data",
            ),
            node(
                shuffle_data,
                inputs=["selected_x", "selected_y", "params:seed"],
                outputs={"shuffled_x": "shuffled_x", "shuffled_y": "shuffled_y"},
                name="shuffle_data",
            ),
            node(
                split_data,
                inputs=[
                    "shuffled_x",
                    "shuffled_y",
                    "params:train_ipc",
                    "params:test_ipc",
                ],
                outputs={
                    "train_x": "train_x",
                    "train_y": "train_y",
                    "test_x": "test_x",
                    "test_y": "test_y",
                },
                name="split_data",
            ),
            node(
                shuffle_data,
                inputs=["train_x", "train_y", "params:seed"],
                outputs={
                    "shuffled_x": "shuffled_train_x",
                    "shuffled_y": "shuffled_train_y",
                },
                name="shuffle_train_data",
            ),
            node(
                normalize_data,
                inputs="shuffled_train_x",
                outputs="shuffled_normalized_train_x",
                name="normalze_train_data",
            ),
            node(
                sort_interleaved,
                inputs=["test_x", "test_y", "params:classes"],
                outputs={"sorted_x": "sorted_test_x", "sorted_y": "sorted_test_y"},
                name="sort_test_data",
            ),
            node(
                normalize_data,
                inputs="sorted_test_x",
                outputs="sorted_normalized_test_x",
                name="normalize_test_data",
            ),
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
                generate_accuracy_curve,
                inputs="autoencoder_history",
                outputs="accuracy_curve",
                name="generate_accuracy_curve",
            )
        ],
        inputs="mnist_data",
        outputs={
            "encoded_test_data": "encoded_test_data",
            "encoded_train_data": "encoded_train_data",
        },
        namespace="data_preprocessing",
    )
