from typing import Sequence, Tuple
import json
import numpy as np
import numpy.typing as npt
import tensorflow as tf  # type: ignore


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
    def __init__(
        self, train, val, train_steps: int, val_steps: int, log_file: str
    ):
        super().__init__()
        self._train = train
        self._val = val
        self._train_steps = train_steps
        self._val_steps = val_steps
        self._log_file = log_file
        open(self._log_file, "w").close()  # Clear contents

    def on_epoch_end(self, epoch: int, logs=None) -> None:
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


def _get_n_rows(args: Sequence[npt.NDArray]) -> int:
    n_rows = len(args[0])
    for arg in args:
        if len(arg) != n_rows:
            raise ValueError("All arguments must be of equal length")
    return n_rows


def shuffle(args: Sequence[npt.NDArray], seed=None) -> Tuple:
    np.random.seed(seed)
    n_rows = _get_n_rows(args)
    new_idx = np.random.choice(n_rows, n_rows, replace=False)
    return tuple(arg[new_idx] for arg in args)


def train_test_split(
    args: Sequence[npt.NDArray], *, train_frac: float
) -> Tuple[Sequence[npt.NDArray], Sequence[npt.NDArray]]:
    n_rows = _get_n_rows(args)
    train_size = int(train_frac * n_rows)
    train = tuple(arg[:train_size] for arg in args)
    test = tuple(arg[train_size:] for arg in args)
    return train, test
