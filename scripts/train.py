import click
import numpy as np
import tensorflow as tf  # type: ignore
from grokking import datasets, models, training


@click.command
@click.argument("metrics_log_file", type=str)
@click.option(
    "--train-frac",
    required=True,
    type=float,
    help="Proportion of data used for training.",
)
@click.option(
    "--shuffle-seed",
    default=23489,
    type=int,
    help="Seed for shuffling dataset.",
)
@click.option(
    "--p",
    default=97,
    type=int,
    help="Prime number p for modular arithmetic operations",
)
@click.option(
    "--train-batch-size",
    default=512,
    type=int,
    help="Maximum batch size (could be smaller for small datasets).",
)
@click.option("--epochs", default=500, type=int, help="Number of epochs.")
@click.option(
    "--steps-per-epoch",
    default=1000,
    type=int,
    help="Number of optimization steps per epoch.",
)
@click.option(
    "--steps-per-execution",
    default=1,
    type=int,
    help="Steps per inner loop execution.",
)
@click.option("--tpu-address", default="", type=str, help="TPU address.")
def run_experiment(
    metrics_log_file: str,
    train_frac: float,
    shuffle_seed: int,
    p: int,
    train_batch_size: int,
    epochs: int,
    steps_per_epoch: int,
    steps_per_execution: int,
    tpu_address: str,
) -> None:
    strategy = training.get_strategy(tpu_address)

    click.echo("Preparing dataset...")
    # Obtain raw dataset and convert to numpy features and targets
    all_equations = datasets.modular_division_dataset(p)
    X = np.array([[equation.x, equation.y] for equation in all_equations])
    y = np.array([equation.res for equation in all_equations])
    n_classes = np.max(y) + 1

    # Shuffle, split and create TF datasets
    X, y = training.shuffle((X, y), seed=shuffle_seed)
    train, val = [
        tf.data.Dataset.from_tensor_slices(ds)
        for ds in training.train_test_split((X, y), train_frac=train_frac)
    ]

    # Print dataset characteristics
    click.echo(f"{n_classes:6d} classes.")
    click.echo(f"{len(all_equations):6d} equations.")
    click.echo(f"{len(train):6d} training examples.")
    click.echo(f"{len(val):6d} validation examples.")

    # Batch and distribute datasets according to strategy
    train_batch_size = min(train_batch_size, len(train))
    val_batch_size = len(val)  # should be able to do in one pass
    train_steps = len(train) // train_batch_size
    val_steps = len(val) // val_batch_size
    assert val_steps == 1  # should be 1 in this case
    dtrain = strategy.experimental_distribute_dataset(
        train.batch(train_batch_size).cache().repeat()
    )
    dval = strategy.experimental_distribute_dataset(
        val.batch(val_batch_size).cache().repeat()
    )

    click.echo("\nStarting training...")
    with strategy.scope():
        model = models.decoder_transformer_classifier(
            2, n_classes, n_classes, 2, 128, 4, 0.0
        )
        evaluator = training.EvaluatorCallback(
            dtrain, dval, train_steps, val_steps, metrics_log_file
        )
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            ),
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=0.001,
                beta_1=0.9,
                beta_2=0.98,
                epsilon=1e-8,
            ),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
            steps_per_execution=steps_per_execution,  # accelerate training
        )
        try:
            model.fit(
                dtrain,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                callbacks=[evaluator],
            )
        except KeyboardInterrupt:
            print("Training interrupted.")

    print(f"Exiting, logs saved to: {metrics_log_file}.")


if __name__ == "__main__":
    run_experiment()
