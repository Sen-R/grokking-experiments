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
    def __init__(self, train, val):
        super().__init__()
        self._history = []
        self._train = train.batch(len(train))
        self._val = val.batch(len(val))

    def on_epoch_end(self, epoch, logs=None):
        print()
        print("Train metrics: ", end="")
        train_metrics = self.model.evaluate(
            self._train, return_dict=True, verbose=2
        )
        print("Val metrics:   ", end="")
        val_metrics = self.model.evaluate(
            self._val, return_dict=True, verbose=2
        )
        print()

        record = {"train": train_metrics, "val": val_metrics}
        self._history.append(record)

    def to_json(self, filename):
        with open(filename, "w") as f:
            json.dump(self._history, f)


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
def run_experiment(
    metrics_log_file: str,
    train_frac: float,
    shuffle_seed: int,
    p: int,
    train_batch_size: int,
    epochs: int,
    steps_per_epoch: int,
) -> None:
    click.echo("Preparing dataset...")
    all_equations = datasets.modular_division_dataset(p)
    n_equations = len(all_equations)

    X = np.array([[equation.x, equation.y] for equation in all_equations])
    y = np.array([equation.res for equation in all_equations])
    np.random.seed(shuffle_seed)
    shuffled_indices = np.random.choice(
        n_equations, n_equations, replace=False
    )
    X = X[shuffled_indices]
    y = y[shuffled_indices]
    n_classes = tf.reduce_max(y).numpy() + 1

    full_dataset = tf.data.Dataset.from_tensor_slices((X, y))
    train_size = int(train_frac * len(all_equations))
    train = full_dataset.take(train_size)
    val = full_dataset.skip(train_size)
    train_batch_size = min(train_batch_size, len(train))
    click.echo(f"{n_classes:6d} classes.")
    click.echo(f"{len(full_dataset):6d} equations.")
    click.echo(f"{len(train):6d} training examples.")
    click.echo(f"{len(val):6d} validation examples.")

    _validate_datasets(train, val, all_equations)

    click.echo("\nStarting training...")
    strategy = get_strategy("")
    with strategy.scope():
        model = models.decoder_transformer_classifier(
            2, n_classes, n_classes, 2, 128, 4, 0.0
        )
        evaluator = EvaluatorCallback(train, val)
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            ),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )
    try:
        model.fit(
            train.batch(train_batch_size).repeat(),
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=[evaluator],
        )
    except KeyboardInterrupt:
        print("Training interrupted.")

    print(f"Saving logs to: {metrics_log_file}.")
    evaluator.to_json(metrics_log_file)


if __name__ == "__main__":
    run_experiment()
