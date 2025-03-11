import io
import sys
from contextlib import contextmanager

import numpy as np


@contextmanager
def mute_outputs():
    """
    With context that temporare mute stdout and stderr.
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
