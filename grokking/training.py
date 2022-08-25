from typing import Sequence, Tuple, Dict, Any
import json
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import numpy.typing as npt
import tensorflow as tf  # type: ignore
from .datasets import Equation


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


def equations_to_arrays(
    equations: Sequence[Equation],
) -> Tuple[npt.NDArray, npt.NDArray]:
    X = np.array([[equation.x, equation.y] for equation in equations])
    y = np.array([equation.res for equation in equations])
    return X, y


def rowwise_cosine_similarity(embedding_matrix: tf.Tensor) -> tf.Tensor:
    """Cosine similarities across rows of input matrix."""
    normalised_matrix, _ = tf.linalg.normalize(embedding_matrix, axis=1)
    pairwise_dot_products = tf.matmul(
        normalised_matrix, normalised_matrix, transpose_b=True
    )
    return pairwise_dot_products


class TrainingLogger(tf.keras.callbacks.Callback):
    def __init__(
        self,
        training_parameters: Dict[str, Any],
        train,
        val,
        results_prefix: str,
    ):
        super().__init__()
        self._train = train.batch(len(train)).cache()
        self._val = val.batch(len(val)).cache()
        results_dir_and_prefix = Path(results_prefix)
        results_dir = results_dir_and_prefix.parent
        results_prefix = results_dir_and_prefix.name

        if not results_dir.is_dir():
            raise ValueError(f"Directory doesn't exist: {results_dir}")

        self.run_dir = (
            results_dir / f"{results_prefix}{datetime.now():%y%m%d%H%M%S}"
        )
        self.run_dir.mkdir()

        json.dump(training_parameters, self.params_file.open("w"))
        self.history_file.open("w")  # clear contents
        self._save_data(self._train, "train.json")
        self._save_data(self._val, "val.json")

    @property
    def params_file(self) -> Path:
        return self.run_dir / "params.json"

    @property
    def history_file(self) -> Path:
        return self.run_dir / "history.json"

    @property
    def data_dir(self) -> Path:
        res = self.run_dir / "data"
        res.mkdir(exist_ok=True)
        return res

    @property
    def checkpoints_dir(self) -> Path:
        res = self.run_dir / "checkpoints"
        res.mkdir(exist_ok=True)
        return res

    def _save_data(self, dataset, filename: str) -> None:
        X, y = [t.numpy().tolist() for t in next(iter(dataset))]
        json.dump([X, y], (self.data_dir / filename).open("w"))

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        print()
        print("Train metrics:", end="")
        train_metrics = self.model.evaluate(
            self._train, return_dict=True, verbose=2, steps=1
        )
        print("Val metrics:  ", end="")
        val_metrics = self.model.evaluate(
            self._val, return_dict=True, verbose=2, steps=1
        )
        layer_weight_norms = {
            t.name: float(tf.norm(t).numpy())
            for t in self.model.trainable_weights
        }
        print(
            "Last layer weight norm:", layer_weight_norms["to_logits/kernel:0"]
        )
        embedding_min_similarity = float(
            tf.reduce_min(
                rowwise_cosine_similarity(self.model.weights[0])
            ).numpy()
        )
        print("Embedding max cos dist:", embedding_min_similarity)

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "layer_weight_norms": layer_weight_norms,
            "embedding_min_similarity": embedding_min_similarity,
        }
        with self.history_file.open("a") as f:
            f.write(json.dumps(record) + "\n")

        if epoch % 10 == 0:
            self.model.save_weights(
                self.checkpoints_dir / f"weights-at-epoch-{epoch:04d}",
                tf.train.CheckpointOptions(experimental_io_device=None),
            )


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
