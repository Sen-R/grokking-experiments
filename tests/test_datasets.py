from itertools import product
from grokking.datasets import (
    binary_operation_dataset,
    modular_division_dataset,
    cubic_polynomial_dataset,
)


class TestBinaryOperationDataset:
    def test_division_dataset(self) -> None:
        p = 5

        def binary_op(x: int, y: int) -> int:
            result = (x * pow(y, -1, p)) % p
            assert (y * result % p) == x
            return result

        equations = binary_operation_dataset(
            fn=binary_op, max_x_y=p - 1, min_y=1
        )
        assert len(equations) == 20  # 5 * 4 combinations
        for equation, (x, y) in zip(equations, product(range(5), range(1, 5))):
            assert equation.x == x
            assert equation.y == y
            assert equation.res == binary_op(equation.x, equation.y)


def test_modular_division_dataset() -> None:
    p = 5
    dataset = modular_division_dataset(p=p)
    assert min(e.x for e in dataset) == 0
    assert max(e.x for e in dataset) == p - 1
    assert min(e.y for e in dataset) == 1
    assert max(e.y for e in dataset) == p - 1
    for x, y, res in dataset:
        assert res * y % p == x


def test_cubic_polynomial_dataset() -> None:
    p = 5
    dataset = cubic_polynomial_dataset(p=p)
    assert min(e.x for e in dataset) == 0
    assert max(e.x for e in dataset) == p - 1
    assert min(e.y for e in dataset) == 0
    assert max(e.y for e in dataset) == p - 1
    for x, y, res in dataset:
        assert res == (x**3 + x * y**2 + y) % p
