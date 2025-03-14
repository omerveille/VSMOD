import json
import math
import time
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
from numpy.testing import assert_array_almost_equal
from ransac_slicer.helper import (
    _nv_to_geo_level,
    flush_timers,
    gradient_central_dif,
    homogenize,
    sample_gauss_sphere,
    sample_half_gauss_sphere,
    time_function,
    timers,  # noqa: F401 # tell the formatter to not remove this variable
)


class HelperTest(unittest.TestCase):
    def test_01_time_function_and_flush_timers(self):
        sleep_time = 0.5
        timers.clear()

        # fake function to trigger the wrapper
        @time_function
        def function_to_time():
            time.sleep(sleep_time)

        # fill timers with a value
        function_to_time()
        self.assertGreater(len(timers), 0)
        self.assertTrue(
            math.isclose(timers[0][1], sleep_time, rel_tol=0.5, abs_tol=0.5)
        )

        # generate a fake timer file
        timers_cpy = deepcopy(timers)
        flushed_file_path = flush_timers()
        self.addCleanup(self.delete_test_file, flushed_file_path)

        # compare file content vs the timers in memory
        file_content = None
        with open(flushed_file_path) as f:
            file_content = json.load(f)

        if isinstance(file_content, list):
            file_content = [tuple(item) for item in file_content]

        flushed_file_path.unlink(missing_ok=True)
        self.assertListEqual(file_content, timers_cpy)

    def test_02__nv_to_geo_level(self):
        self.assertEqual(_nv_to_geo_level(12), 0)
        self.assertEqual(_nv_to_geo_level(13), 1)
        self.assertEqual(_nv_to_geo_level(43), 2)
        self.assertEqual(_nv_to_geo_level(163), 3)

    def test_03_sample_gauss_sphere(self):
        with self.assertRaises(ValueError):
            sample_gauss_sphere(n_vertices=1, radius=0)

        self.assertGreaterEqual(sample_gauss_sphere(n_vertices=1).shape[0], 1)
        self.assertGreaterEqual(sample_gauss_sphere(n_vertices=13).shape[0], 13)
        self.assertGreaterEqual(sample_gauss_sphere(n_vertices=43).shape[0], 43)
        self.assertGreaterEqual(sample_gauss_sphere(n_vertices=163).shape[0], 163)

    def test_04_sample_half_gauss_sphere(self):
        self.assertGreaterEqual(sample_half_gauss_sphere(n_vertices=1).shape[0], 1)
        self.assertGreaterEqual(sample_half_gauss_sphere(n_vertices=13).shape[0], 13)
        self.assertGreaterEqual(sample_half_gauss_sphere(n_vertices=43).shape[0], 43)
        self.assertGreaterEqual(sample_half_gauss_sphere(n_vertices=163).shape[0], 163)

    def test_05_gradient_central_dif(self):
        test_array = np.array(
            [i for i in range(10)] + [i for i in range(8, -1, -1)], dtype=np.float64
        )
        gradient_diff = gradient_central_dif(test_array)
        expected_result = np.array(
            [
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                0.0,
            ],
            dtype=np.float64,
        )
        assert_array_almost_equal(
            gradient_diff,
            expected_result,
        )

    def test_06_homogenize(self):
        one_dim_shape = homogenize(np.random.random(size=3)).shape
        two_dim_shape = homogenize(np.random.random(size=30).reshape(10, 3)).shape

        self.assertTupleEqual(one_dim_shape, (4,))
        self.assertTupleEqual(two_dim_shape, (10, 4))

    def delete_test_file(self, path: Path):
        path.unlink(missing_ok=True)
