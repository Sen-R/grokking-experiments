"""Components for loading and analysing experiment results."""


from typing import Dict, Any, Tuple
from pathlib import Path
import json
import pandas as pd  # type: ignore
import numpy as np
import numpy.typing as npt
import tensorflow as tf  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.colors import hsv_to_rgb  # type: ignore
from . import models


_param_defaults = {
    "dataset": "modular_division",
    "train_frac": 0.5,
    "shuffle_seed": 23489,
    "p": 97,
    "model_name": "transformer",
    "layers": 2,
    "width": 128,
    "heads": 4,
    "dropout": 0.0,
    "embedding_weights": "learned",
    "hidden_layers": [200, 200, 30],
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "beta_1": 0.9,
    "beta_2": 0.98,
    "epsilon": 1e-8,
    "train_batch_size": 512,
    "epochs": 500,
    "steps_per_epoch": 1000,
    "steps_per_execution": 1,
}


def _validate_parameters_and_populate_defaults(params: Dict[str, Any]) -> None:
    for name in params:
        if name not in _param_defaults:
            raise ValueError(f"Unrecognised parameter: {name}")
    for name, default_value in _param_defaults.items():
        if name not in params:
            params[name] = default_value


class Run:
    """Object to host training history, data and model checkpoints from a
    single experimental run."""

    def __init__(self, dirname: str):
        self._dirpath = Path(dirname)
        if not self._dirpath.is_dir():
            raise ValueError(
                f"Directory doesn't exist or isn't a directory: {dirname}"
            )

        self._load_parameters()
        self._load_history()
        self._load_train_val_datasets()

    def _load_parameters(self) -> None:
        self.params = json.load((self._dirpath / "params.json").open("r"))
        _validate_parameters_and_populate_defaults(self.params)

    def _load_history(self) -> None:
        self.history = pd.read_json(
            self._dirpath / "history.json", lines=True
        ).set_index("epoch")

    def _load_train_val_datasets(self) -> None:
        Xt, yt = json.load((self._dirpath / "data/train.json").open("r"))
        Xv, yv = json.load((self._dirpath / "data/val.json").open("r"))
        self.train = np.array(Xt), np.array(yt)
        self.val = np.array(Xv), np.array(yv)

        self.n_input_tokens = (
            max(np.max(self.train[0]), np.max(self.val[0])) + 1
        )  # including 0 means we need to add 1
        self.n_output_tokens = (
            max(np.max(self.train[1]), np.max(self.val[1])) + 1
        )

        # For the datasets we have implemented so far, the following
        # assertions should hold true. Will need to remove for more
        # general datasets (e.g. non-modular add)
        assert self.n_input_tokens == self.params["p"]
        assert self.n_output_tokens == self.params["p"]

    def model_for_epoch(self, epoch: int) -> tf.keras.Model:
        model = models.build(
            2, self.n_input_tokens, self.n_output_tokens, self.params
        )
        checkpoint = tf.train.Checkpoint(model)
        ckpt_path = str(
            self._dirpath / f"checkpoints/weights-at-epoch-{epoch:04d}"
        )
        checkpoint.restore(ckpt_path).expect_partial()
        return model

    def predictions_for_epoch(
        self, epoch: int, split: str
    ) -> Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
        if split not in ("train", "val"):
            raise ValueError(f"Invalid split: {split}")
        model = self.model_for_epoch(epoch)
        X = getattr(self, split)[0]
        logits = model(tf.constant(X)).numpy()
        preds = np.argmax(logits, axis=-1)
        return X, preds

    def learning_curves(
        self, metric: str, ax=None, xlim=None, ylim=None, ylabel=None
    ) -> None:
        ax = plt.gca() if ax is None else ax
        self.history.train.str[metric].plot(ax=ax)
        self.history.val.str[metric].plot(ax=ax)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_ylabel(metric if ylabel is None else ylabel)


def _dataset_to_matrix(
    inputs: npt.NDArray[np.int_],
    values: npt.NDArray[np.int_],
    missing_value: npt.ArrayLike = -1,
) -> npt.NDArray:
    assert np.min(inputs) >= 0
    shape = (np.max(inputs[:, 0]) + 1, np.max(inputs[:, 1]) + 1)
    mat = np.tile(missing_value, np.prod(shape)).reshape([*shape, -1])
    for (x, y), value in zip(inputs, values):
        mat[x, y] = value
    return mat.squeeze()


def visualise(
    inputs: npt.NDArray[np.int_],
    res: npt.NDArray[np.int_],
    permute_token_orders: bool,
    seed=None,
    ax=None,
    **kwargs,
) -> None:
    n_input_tokens = np.max(inputs) + 1
    n_output_tokens = np.max(inputs) + 1

    if permute_token_orders:
        np.random.seed(seed)
        input_perm = np.random.permutation(n_input_tokens)
        output_perm = np.random.permutation(n_output_tokens)
        inputs = input_perm[inputs]
        res = output_perm[res]

    hsv = np.ones([len(res), 3])
    hsv[:, 0] = res / n_output_tokens
    rgb = hsv_to_rgb(hsv)

    mat = _dataset_to_matrix(
        inputs, rgb, missing_value=np.array([1.0, 1.0, 1.0])
    )
    ax = plt.gca() if ax is None else ax
    ax.imshow(mat)
