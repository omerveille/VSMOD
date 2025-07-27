import unittest
from math import isclose
from pathlib import Path
import inspect
import sys
import importlib
import time

import numpy as np
from numpy.testing import assert_array_almost_equal
from ransac_slicer.jit_compiled_functions import (
    atomic_exchange,
    numba_close,
    numba_cross,
    numba_distance,
    numba_filter_points,
    numba_fit_3_points_cylinder,
    numba_mark_selected_inliers,
    numba_random_indices,
    numba_fit_cylinder_ransac,
)
from .test_utils import generate_points, get_resources_path, load_object_from_path


class JitCompiledFunctionsTest(unittest.TestCase):
    def test_01_numba_random_indices(self):
        np.random.seed(42)
        for _ in range(100):
            self.assertEqual(np.unique(numba_random_indices(100)).shape[0], 3)

    def test_02_atomic_exchange(self):
        np.random.seed(42)
        for _ in range(100):
            flag = np.random.randint(low=0, high=1000, size=(1,), dtype=np.int64)

            expected_old_flag_value = flag[0]
            old_flag_value = atomic_exchange(flag, 42)

            self.assertEqual(old_flag_value, expected_old_flag_value)
            self.assertEqual(flag[0], 42)

    def test_03_numba_close(self):
        for i in range(64):
            value = 1**-i
            self.assertEqual(numba_close(value, 0.0), isclose(value, 0.0))

    def test_04_numba_cross(self):
        np.random.seed(42)
        for _ in range(100):
            a = np.random.random(3)
            b = np.random.random(3)
            assert_array_almost_equal(np.cross(a, b), numba_cross(a, b))

    def test_05_numba_fit_3_points_cylinder(self):
        expected_error_result = (
            np.zeros(3, dtype=np.float64),
            -1.0,
            np.zeros(3, dtype=np.float64),
        )

        # Case error direction vector is null
        a = np.array([0, 0, 1], dtype=np.float64)
        b = np.array([0, 1, 0], dtype=np.float64)
        c = np.array([1, 0, 1], dtype=np.float64)

        d = np.zeros(shape=(3,), dtype=np.float64)

        result = numba_fit_3_points_cylinder(a, b, c, d)

        for result_expect, result_ in zip(expected_error_result, result):
            if isinstance(result_expect, np.ndarray):
                assert_array_almost_equal(result_expect, result_)
            else:
                self.assertEqual(isclose(result_expect, -1.0), isclose(result_, -1.0))

        # Case error the points cannot define a plane
        a = np.zeros(shape=(3,), dtype=np.float64)
        b = np.zeros(shape=(3,), dtype=np.float64)
        c = np.zeros(shape=(3,), dtype=np.float64)

        result = numba_fit_3_points_cylinder(a, b, c, d)

        for result_expect, result_ in zip(expected_error_result, result):
            if isinstance(result_expect, np.ndarray):
                assert_array_almost_equal(result_expect, result_)
            else:
                self.assertEqual(isclose(result_expect, -1.0), isclose(result_, -1.0))

        # Case normal usage
        a = np.array([0, 0, 1], dtype=np.float64)
        b = np.array([0, 1, 0], dtype=np.float64)
        c = np.array([1, 0, 1], dtype=np.float64)

        d = np.array([1, 1, 1], dtype=np.float64)
        result = numba_fit_3_points_cylinder(a, b, c, d)

        for result_expect, result_ in zip(expected_error_result, result):
            if isinstance(result_expect, np.ndarray):
                for err_val, res_val in zip(result_expect, result_):
                    self.assertFalse(isclose(err_val, res_val))

            else:
                self.assertEqual(
                    isclose(result_expect, -1.0), not isclose(result_, -1.0)
                )

    def test_06_numba_distance(self):
        # Test with points on the surface of the cylinder
        points, center, radius, direction = generate_points(
            number_of_points=300, radius=1
        )

        distance_result = numba_distance(points, center, radius, direction)
        assert_array_almost_equal(
            distance_result,
            np.zeros_like(distance_result),
        )

        # Test with points outside the cylinder (distance 1)
        points, _, _, _ = generate_points(
            number_of_points=300, radius=1, x_y_noise_range=(2, 2)
        )
        distance_result = numba_distance(points, center, radius, direction)
        assert_array_almost_equal(
            numba_distance(points, center, radius, direction),
            np.ones_like(distance_result),
        )

        # Test with points inside the cylinder (on the axis, so distance 0)
        points = points, _, _, _ = generate_points(
            number_of_points=300, radius=1, x_y_noise_range=(0, 0)
        )
        distance_result = numba_distance(points, center, radius, direction)
        assert_array_almost_equal(distance_result, -np.ones_like(distance_result))

    def test_07_numba_mark_selected_inliers(self):
        inlier_threshold = 0.25
        inlier_points, center, radius, direction = generate_points(
            number_of_points=300, radius=1
        )
        outlier_1_points, _, _, _ = generate_points(number_of_points=300, radius=1.26)
        outlier_2_points, _, _, _ = generate_points(number_of_points=300, radius=0.74)

        self.assertTrue(
            numba_mark_selected_inliers(
                inlier_points, inlier_threshold, center, radius, direction
            ).all()
        )
        self.assertFalse(
            numba_mark_selected_inliers(
                outlier_1_points, inlier_threshold, center, radius, direction
            ).any()
        )
        self.assertFalse(
            numba_mark_selected_inliers(
                outlier_2_points, inlier_threshold, center, radius, direction
            ).any()
        )

    def test_08_numba_filter_points(self):
        ok_points, center, _, _ = generate_points(
            number_of_points=300, radius=1, z_range=(0, 0)
        )
        ko_1_points, _, _, _ = generate_points(
            number_of_points=300, radius=1.26, z_range=(0, 0)
        )
        ko_2_points, _, _, _ = generate_points(
            number_of_points=300, radius=0.74, z_range=(0, 0)
        )
        self.assertTrue(
            numba_filter_points(ok_points, center, 0.9, 1.1).shape[0] == 300
        )
        self.assertTrue(
            numba_filter_points(ko_1_points, center, 0.9, 1.1).shape[0] == 0
        )
        self.assertTrue(
            numba_filter_points(ko_2_points, center, 0.9, 1.1).shape[0] == 0
        )

    def test_09_numba_fit_cylinder_ransac(self):
        # This chunk of code is used to import the function

        salt = str(time.time()).replace(".", "")
        filename = f"test_numba_fit_cylinder_ransac_{salt}"
        function_name = f"numba_fit_cylinder_ransac_{salt}"
        function_file_path = Path(__file__).parent.joinpath(f"{filename}.py").absolute()

        self.addCleanup(self.delete_test_file, function_file_path)
        function_source_code = inspect.getsource(numba_fit_cylinder_ransac)
        # Replace the decorator in order to make the function outputs reproductible
        function_source_code = function_source_code.replace(
            "@njit(nogil=True, parallel=True, cache=True)",
            "@njit(nogil=False, parallel=False, cache=False)",
        )
        # Add seeding
        function_source_code = function_source_code.replace(
            "# np.random.seed(42)", "np.random.seed(42)"
        )
        # Make the function name unique to make sure it does not import cached version of it
        function_source_code = function_source_code.replace(
            "numba_fit_cylinder_ransac", function_name
        )

        import_code = """from ransac_slicer.jit_compiled_functions import (
    numba_fit_3_points_cylinder,
    numba_random_indices,
    numba_mark_selected_inliers,
    atomic_exchange,
)
import numpy as np
from numba import njit, prange
"""
        function_file_path.touch()

        with open(function_file_path, "w") as f:
            f.write(import_code)
            f.write(function_source_code)

        spec = importlib.util.spec_from_file_location(filename, function_file_path)
        module = importlib.util.module_from_spec(spec)
        self.addCleanup(self.clean_module_after_test, module.__name__)
        spec.loader.exec_module(module)

        reproductible_numba_fit_cylinder_ransac: numba_fit_cylinder_ransac = getattr(
            module, function_name
        )

        ressource_test_dir = get_resources_path()

        p = load_object_from_path(
            ressource_test_dir.joinpath("jit_compiled_functions").joinpath("p.npy")
        )
        axis = load_object_from_path(
            ressource_test_dir.joinpath("jit_compiled_functions").joinpath("axis.npy")
        )

        # Test using previously known data
        basis, inlier, percent = reproductible_numba_fit_cylinder_ransac(
            p=p,
            axis=axis,
            nb_test_min=20000,
            nb_test_max=20000,
            sufficient_pct_inl=0.75,
            r_min=0.54,
            r_max=1.62,
            err=0.324,
        )

        expected_basis = load_object_from_path(
            ressource_test_dir.joinpath("jit_compiled_functions").joinpath(
                "expected_basis.npy"
            )
        )
        expected_inlier = load_object_from_path(
            ressource_test_dir.joinpath("jit_compiled_functions").joinpath(
                "expected_inlier.npy"
            )
        )
        expected_inlier_pct = 0.8090909090909091

        assert_array_almost_equal(basis, expected_basis)
        assert_array_almost_equal(inlier, expected_inlier)
        self.assertTrue(isclose(percent, expected_inlier_pct))

        # Tests with synthetic data
        def fit_cylinder(x_y_noise_range: tuple[float, float]):
            p, _, _, _ = generate_points(
                number_of_points=300,
                radius=1,
                x_y_noise_range=x_y_noise_range,
                z_range=(-1, 1),
            )
            axis = np.array([0, 0, 1], dtype=np.float64)

            basis, _, rate = reproductible_numba_fit_cylinder_ransac(
                p=p,
                axis=axis,
                nb_test_min=20000,
                nb_test_max=20000,
                sufficient_pct_inl=0.75,
                r_min=0.0,
                r_max=2.0,
                err=0.3,
            )
            _, radius, _ = numba_fit_3_points_cylinder(*basis, axis)
            relative_difference_of_radius = abs(1 - radius) / ((1 + radius) / 2) * 100
            return rate * 100, relative_difference_of_radius

        expected_radius_relative_difference_threshold = [20] * 2 + [5] * 6
        expected_inlier_percent_threshold = [40] * 4 + [50, 60, 70, 90]
        for scale, radius_threshold_diff, inlier_percent_threshold in zip(
            range(0, 9, 1),
            expected_radius_relative_difference_threshold,
            expected_inlier_percent_threshold,
        ):
            scale *= 0.1
            percent, relative_difference_of_radius = fit_cylinder((scale, 2.0 - scale))
            self.assertTrue(relative_difference_of_radius < radius_threshold_diff)
            self.assertTrue(percent > inlier_percent_threshold)

        function_file_path.unlink()

    def delete_test_file(self, path: Path):
        path.unlink(missing_ok=True)

    def clean_module_after_test(self, module_name):
        sys.modules.pop(module_name, None)
