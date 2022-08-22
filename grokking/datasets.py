from typing import NamedTuple, Callable, List
from .utils import modinv


class Equation(NamedTuple):
    x: int
    y: int
    res: int


def binary_operation_dataset(
    fn: Callable[[int, int], int], max_x_y: int, min_y: int = 0
) -> List[Equation]:
    return [
        Equation(x, y, fn(x, y))
        for x in range(max_x_y + 1)
        for y in range(min_y, max_x_y + 1)
    ]


def _modular_div_op(p: int) -> Callable[[int, int], int]:
    def modular_div(x: int, y: int) -> int:
        return (x * modinv(y, p)) % p

    return modular_div


def modular_division_dataset(p: int) -> List[Equation]:
    max_x_y = p - 1
    min_y = 1
    return binary_operation_dataset(_modular_div_op(p), max_x_y, min_y)


def cubic_polynomial_dataset(p: int) -> List[Equation]:
    max_x_y = p - 1
    min_y = 0
    return binary_operation_dataset(
        (lambda x, y: (x**3 + x * y**2 + y) % p), max_x_y, min_y
    )


_known_datasets = {
    "modular_division": modular_division_dataset,
    "cubic_polynomial": cubic_polynomial_dataset,
}


def load(name: str, **kwargs) -> List[Equation]:
    return _known_datasets[name](**kwargs)
