import io
import sys
from contextlib import contextmanager
from typing import Union
from numpy.testing import assert_almost_equal
from math import isclose
from pathlib import Path
import slicer

import numpy as np


@contextmanager
def mute_outputs():
    """
    With context that temporary mute stdout and stderr.
    """
    saved_stdout = sys.stdout
    saved_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = saved_stdout
        sys.stderr = saved_stderr


def generate_points(
    number_of_points: int,
    radius: float,
    x_y_noise_range: tuple[float, float] = [1.0, 1.0],
    z_range: tuple[float, float] = (-1, 1),
    center: np.ndarray = np.zeros(shape=(3,), dtype=np.float64),
    direction: np.ndarray = np.array([0, 0, 1], dtype=np.float64),
):
    np.random.seed(42)
    angles = np.random.uniform(0, 2 * np.pi, number_of_points)

    radius_random_variation = np.random.uniform(*x_y_noise_range, number_of_points)

    x = (center[0] + radius * np.cos(angles)) * radius_random_variation
    y = (center[1] + radius * np.sin(angles)) * radius_random_variation
    z = np.random.uniform(*z_range, number_of_points)

    return np.column_stack((x, y, z)), center, radius, direction


def get_resources_path() -> Path:
    return (
        Path(__file__).parent.parent.joinpath("Resources").joinpath("Test").absolute()
    )


def load_object_from_path(path: Path) -> object:
    if path.name.endswith(".npy"):
        return np.load(path)
    else:
        return slicer.util.loadVolume(str(path))


def custom_iterable_assertion(list_1: Union[list, tuple], list_2: Union[list, tuple]):
    if list_1 is list_2:
        return
    assert len(list_1) == len(list_2)
    for item_1, item_2 in zip(list_1, list_2):
        assert type(item_1) is type(item_2)
        if type(item_1) in [list, tuple]:
            custom_iterable_assertion(item_1, item_2)
        elif isinstance(item_1, np.ndarray):
            assert_almost_equal(item_1, item_2)
        elif type(item_1) in [float, int, np.float64]:
            assert isclose(item_1, item_2)
        else:
            assert item_1 == item_2
