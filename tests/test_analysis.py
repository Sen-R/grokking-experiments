import numpy as np
from numpy.testing import assert_array_equal
from grokking.analysis import _dataset_to_matrix


inputs = np.array([[0, 0], [3, 0], [3, 2]])
res = np.array([0, 3, 4])


def test_dataset_to_matrix() -> None:
    mat = _dataset_to_matrix(inputs, res, missing_value=-2)
    print(mat)
    assert_array_equal(mat.shape, [4, 3])
    assert_array_equal(
        mat, [[0, -2, -2], [-2, -2, -2], [-2, -2, -2], [3, -2, 4]]
    )
