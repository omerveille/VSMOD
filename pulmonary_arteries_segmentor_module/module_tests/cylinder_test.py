import unittest
from ransac_slicer.cylinder import Cylinder, dist_to_branch, closest_branch
import numpy as np
from numpy.testing import assert_array_almost_equal
from .test_utils import generate_points
from math import isclose


class CylinderTest(unittest.TestCase):
    def test_01_Cylinder___init__(self):
        with self.assertRaises(ValueError):
            Cylinder(radius=-1)
        with self.assertRaises(ValueError):
            Cylinder(direction=np.zeros(shape=(3,), dtype=np.float64))

        Cylinder()

    def test_02_Cylinder_copy(self):
        cyl_a = Cylinder()
        cyl_b = cyl_a.copy()

        self.assertIsNot(cyl_a, cyl_b)
        assert_array_almost_equal(cyl_a.center, cyl_b.center)
        self.assertEqual(cyl_a.radius, cyl_b.radius)
        assert_array_almost_equal(cyl_a.direction, cyl_b.direction)
        self.assertEqual(cyl_a.height, cyl_b.height)

    def test_03_Cylinder_distance(self):
        # Test with points on the surface of the cylinder
        points, center, radius, direction = generate_points(
            number_of_points=300, radius=1
        )

        cylinder = Cylinder(center=center, radius=radius, direction=direction)
        distance_result = cylinder.distance(points)
        assert_array_almost_equal(distance_result, np.zeros_like(distance_result))

        # Test with points outside the cylinder (distance 1)
        points, _, _, _ = generate_points(
            number_of_points=300, radius=1, x_y_noise_range=(2, 2)
        )
        distance_result = cylinder.distance(points)
        assert_array_almost_equal(distance_result, np.ones_like(distance_result))

        # Test with points inside the cylinder (on the axis, so distance 0)
        points = points, _, _, _ = generate_points(
            number_of_points=300, radius=1, x_y_noise_range=(0, 0)
        )
        distance_result = cylinder.distance(points)
        assert_array_almost_equal(distance_result, -np.ones_like(distance_result))

    def test_04_Cylinder_select_inliers(self):
        inlier_threshold = 0.25
        inlier_points, center, radius, direction = generate_points(
            number_of_points=300, radius=1
        )
        outlier_1_points, _, _, _ = generate_points(number_of_points=300, radius=1.26)
        outlier_2_points, _, _, _ = generate_points(number_of_points=300, radius=0.74)
        cylinder = Cylinder(center=center, radius=radius, direction=direction)

        self.assertTrue(
            cylinder.select_inliers(inlier_points, inlier_threshold).shape[0]
            == inlier_points.shape[0]
        )
        self.assertTrue(
            cylinder.select_inliers(outlier_1_points, inlier_threshold).shape[0] == 0
        )
        self.assertTrue(
            cylinder.select_inliers(outlier_2_points, inlier_threshold).shape[0] == 0
        )

    def test_05_Cylinder_fix_center(self):
        inlier_points, center, radius, direction = generate_points(
            number_of_points=1000, radius=1
        )
        cylinder = Cylinder(center=center, radius=radius, direction=direction)
        cylinder.fix_center(inlier_points)
        assert_array_almost_equal(cylinder.center, center, decimal=2)

    def test_06_Cylinder_fix_height(self):
        inlier_points, center, radius, direction = generate_points(
            number_of_points=1000, radius=1
        )
        cylinder = Cylinder(center=center, radius=radius, direction=direction)
        cylinder.fix_height(inlier_points)

        # The value expected is 1.5, because the fix height function exclude 25 % of the farthest points
        self.assertTrue(isclose(1.5, cylinder.height, rel_tol=0.1, abs_tol=0.1))

    def test_07_Cylinder_refine(self):
        inlier_points, center, radius, direction = generate_points(
            number_of_points=1000, radius=1
        )
        cylinder = Cylinder(center=center, radius=radius, direction=direction)
        cylinder.direction = np.array([0.25, 0.25, 1], dtype=np.float64)
        cylinder.refine(inlier_points)
        assert_array_almost_equal(cylinder.direction, direction)
        assert_array_almost_equal(cylinder.center, center, decimal=3)
        self.assertTrue(isclose(cylinder.radius, radius, abs_tol=0.01, rel_tol=0.01))

    def test_08_Cylinder_is_redundant(self):
        end = np.array([0, 0, 1], dtype=np.float64)
        points = np.linspace(np.zeros(shape=(3,), dtype=np.float64), end, 10)
        branch = [
            Cylinder(center=point, direction=end, height=1, radius=1)
            for point in points
        ]

        self.assertTrue(branch[0].is_redundant(branch))

        out_of_branch_start = np.array([0, 0, 10], dtype=np.float64)
        out_of_branch_cyls: list[Cylinder] = [
            Cylinder(center=point, direction=end, height=1, radius=1)
            for point in np.linspace(
                out_of_branch_start, np.zeros_like(out_of_branch_start), 101
            )
        ]

        number_of_redundant_cyl = np.sum(
            [cyl.is_redundant(branch) for cyl in out_of_branch_cyls]
        )
        self.assertTrue(number_of_redundant_cyl == 11)

    def test_09_dist_to_branch(self):
        out_of_branch_points, center, radius, direction = generate_points(
            number_of_points=300, radius=1
        )

        end = np.array([0, 0, 1], dtype=np.float64)
        branch_points = np.linspace(np.array([0, 0, -1], dtype=np.float64), end, 10)
        branch = [
            Cylinder(center=point, direction=end, height=1, radius=1)
            for point in branch_points
        ]

        distances = np.asarray(
            [dist_to_branch(point, branch)[0] for point in out_of_branch_points],
            dtype=np.float64,
        )
        assert_array_almost_equal(distances, np.ones_like(distances))

    def test_10_closest_branch(self):
        branches = [
            list(
                Cylinder(center=np.array([start, 0, end], dtype=np.float64))
                for end in [1, 2]
            )
            for start in range(-2, 3)
        ]

        points = [np.array([start, 0, 0], dtype=np.float64) for start in range(-2, 3)]
        for idx, point in enumerate(points):
            _, _, idx_closest_branch, _ = closest_branch(point, branches)
            self.assertTrue(idx == idx_closest_branch)
