import click
import json
import numpy as np
import tensorflow as tf  # type: ignore
from grokking import datasets, models


def get_strategy(tpu_address="local"):
    """Return TPU strategy if possible, else default strategy."""
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu_address
        )
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
    except (tf.errors.NotFoundError, ValueError):
        print("No TPU found, backing off to default strategy.")
        strategy = tf.distribute.get_strategy()
    print("All devices", tf.config.list_logical_devices(), "\n\n")
    return strategy


class EvaluatorCallback(tf.keras.callbacks.Callback):
    def __init__(self, train, val, train_steps, val_steps, log_file):
        super().__init__()
        self._train = train
        self._val = val
        self._train_steps = train_steps
        self._val_steps = val_steps
        self._log_file = log_file
        open(self._log_file, "w").close()  # Clear contents

    def on_epoch_end(self, epoch, logs=None):
        print()
        print("Train metrics: ", end="")
        train_metrics = self.model.evaluate(
            self._train, return_dict=True, verbose=2, steps=self._train_steps
        )
        print("Val metrics:   ", end="")
        val_metrics = self.model.evaluate(
            self._val, return_dict=True, verbose=2, steps=self._val_steps
        )
        print()

        record = {"train": train_metrics, "val": val_metrics}
        with open(self._log_file, "a") as f:
            f.write(json.dumps(record) + "\n")


def _validate_datasets(train, val, all_equations):
    train_X = {tuple(e[0].numpy()) for e in train}
    val_X = {tuple(e[0].numpy()) for e in val}
    all_X = {(e.x, e.y) for e in all_equations}
    assert train_X & val_X == set()
    assert train_X | val_X == all_X


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
    strategy = get_strategy(tpu_address)

    click.echo("Preparing dataset...")
    # Obtain raw dataset
    all_equations = datasets.modular_division_dataset(p)
    n_equations = len(all_equations)

    # Shuffle and convert to np arrays
    np.random.seed(shuffle_seed)
    shuffled_idx = np.random.choice(n_equations, n_equations, replace=False)
    X = np.array([[equation.x, equation.y] for equation in all_equations])[
        shuffled_idx
    ]
    y = np.array([equation.res for equation in all_equations])[shuffled_idx]
    n_classes = np.max(y) + 1

    # Split and create TF datasets
    train_size = int(train_frac * len(all_equations))
    train = tf.data.Dataset.from_tensor_slices(
        (X[:train_size], y[:train_size])
    )
    val = tf.data.Dataset.from_tensor_slices((X[train_size:], y[train_size:]))
    click.echo(f"{n_classes:6d} classes.")
    click.echo(f"{len(all_equations):6d} equations.")
    click.echo(f"{len(train):6d} training examples.")
    click.echo(f"{len(val):6d} validation examples.")
    _validate_datasets(train, val, all_equations)

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
        evaluator = EvaluatorCallback(
            dtrain, dval, train_steps, val_steps, metrics_log_file
        )
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            ),
            optimizer=tf.keras.optimizers.Adam(),
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
