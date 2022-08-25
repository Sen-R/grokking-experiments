import pytest
import numpy as np
from numpy.testing import assert_array_equal
from grokking.datasets import Equation
from grokking.training import (
    shuffle,
    train_test_split,
    _get_n_rows,
    equations_to_arrays,
)


np.random.seed(298374)
X = np.random.normal(size=[5, 2])
y = np.random.normal(size=[5])
y_wrong_size = np.random.normal(size=[4])


class TestGetNRows:
    def test_functionality(self) -> None:
        n_rows = _get_n_rows((X, y))
        assert n_rows == 5

    def test_raises_if_misaligned(self) -> None:
        with pytest.raises(ValueError):
            _get_n_rows((X, y_wrong_size))


class TestShuffle:
    def test_functionality(self) -> None:
        Xs, ys = shuffle((X, y), seed=98234)
        assert {tuple(row) for row in X} == {tuple(row) for row in Xs}
        assert {*y} == {*ys}

    @pytest.mark.skip
    def test_alignment(self) -> None:
        raise NotImplementedError


class TestTrainTestSplit:
    def test_functionality(self) -> None:
        (Xt, yt), (Xv, yv) = train_test_split((X, y), train_frac=0.6)
        assert_array_equal(Xt, X[:3])
        assert_array_equal(yt, y[:3])
        assert_array_equal(Xv, X[3:])
        assert_array_equal(yv, y[3:])


class TestEquationsToArrays:
    def test_functionality(self) -> None:
        equations = [
            Equation(1, 1, 2),
            Equation(2, 2, 4),
        ]
        X, y = equations_to_arrays(equations)
        assert_array_equal(X, [[1, 1], [2, 2]])
        assert_array_equal(y, [2, 4])
