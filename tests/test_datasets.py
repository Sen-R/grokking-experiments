from itertools import product
import pytest
from grokking.datasets import binary_operation_dataset, load


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


@pytest.mark.parametrize(
    "name,p,max_x_y,min_y,test",
    [
        [
            "modular_division",
            5,
            4,
            1,
            (lambda p, x, y, res: (res * y % p) - x),
        ],
        [
            "cubic_polynomial",
            5,
            4,
            0,
            (lambda p, x, y, res: (x**3 + x * y**2 + y) % p - res),
        ],
        [
            "modular_addition",
            5,
            4,
            0,
            (lambda p, x, y, res: (x + y) % p - res),
        ],
    ],
)
def test_modular_division_dataset(
    name: str, p: int, max_x_y: int, min_y: int, test
) -> None:
    dataset = load(name, p=p)
    assert min(e.x for e in dataset) == 0
    assert max(e.x for e in dataset) == max_x_y
    assert min(e.y for e in dataset) == min_y
    assert max(e.y for e in dataset) == max_x_y
    for x, y, res in dataset:
        assert test(p, x, y, res) == 0


class TestLoad:
    @pytest.mark.parametrize(
        "name,p,length",
        [("modular_division", 5, 20), ("cubic_polynomial", 5, 25)],
    )
    def test_call(self, name: str, p: int, length: int) -> None:
        equations = load(name, p=p)
        assert len(equations) == length
