from typing import Sequence, Tuple, Dict, Any
import json
import os
from pathlib import Path
import numpy as np
import numpy.typing as npt
import tensorflow as tf  # type: ignore


def _get_strategy():
    """Return TPU strategy if TPU_ADDRESS set, else default strategy."""
    tpu_address = os.getenv("TPU_ADDRESS")
    if tpu_address is not None:
        print("Trying to set up TPU strategy...")
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu_address
        )
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
    else:
        strategy = tf.distribute.get_strategy()
    print("All devices:", tf.config.list_logical_devices(), "\n\n")
    return strategy


strategy = _get_strategy()  # Set up once when module is loaded


class TrainingLogger(tf.keras.callbacks.Callback):
    def __init__(
        self,
        training_parameters: Dict[str, Any],
        train,
        val,
        results_dir: str,
    ):
        super().__init__()
        self._train = train.batch(len(train)).cache()
        self._val = val.batch(len(val)).cache()
        self._results_dir = Path(results_dir)

        if not self._results_dir.is_dir():
            raise ValueError(f"Directory doesn't exist: {results_dir}")

        json.dump(training_parameters, self.params_file.open("w"))
        self.history_file.open("w")  # clear contents

    @property
    def params_file(self) -> Path:
        return self._results_dir / "params.json"

    @property
    def history_file(self) -> Path:
        return self._results_dir / "history.json"

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        print()
        print("Train metrics: ", end="")
        train_metrics = self.model.evaluate(
            self._train, return_dict=True, verbose=2, steps=1
        )
        print("Val metrics:   ", end="")
        val_metrics = self.model.evaluate(
            self._val, return_dict=True, verbose=2, steps=1
        )
        print()

        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        with self.history_file.open("a") as f:
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
