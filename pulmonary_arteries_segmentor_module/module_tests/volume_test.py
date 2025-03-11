import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal
from ransac_slicer.volume import Volume


class VolumeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Default variables
        """
        cls.ijk_to_ras = np.array(
            [
                [1.0, 0.0, 0.0, 10.0],
                [0.0, 2.0, 0.0, 20.0],
                [0.0, 0.0, 3.0, 30.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        cls.volume = Volume(ijk_to_ras=cls.ijk_to_ras)

    def test_01_Volume___init___error_cases(self):
        with self.assertRaises(ValueError):
            Volume(data=np.zeros(shape=(0, 0)))

        with self.assertRaises(ValueError):
            Volume(ijk_to_ras=np.zeros(shape=(0, 0)))

        Volume()

    def test_02_Volume_transf_ijk_to_ras(self):
        ijk_to_ras = self.ijk_to_ras
        vol = self.volume

        test_point_ijk = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        expected_ras = (np.array([1.0, 2.0, 3, 1.0], dtype=np.float64) @ ijk_to_ras.T)[
            :3
        ]

        assert_array_almost_equal(vol.transf_ijk_to_ras(test_point_ijk), expected_ras)

        assert_array_almost_equal(
            vol.transf_ijk_to_ras(np.zeros(shape=(3,), dtype=np.float64)),
            ijk_to_ras[:3, 3],
        )

        vol = Volume(ijk_to_ras=np.eye(4, dtype=np.float64))
        assert_array_almost_equal(
            vol.transf_ijk_to_ras(np.array([5.0, 5.0, 5.0], dtype=np.float64)),
            np.array([5.0, 5.0, 5.0], dtype=np.float64),
        )

    def test_03_Volume_transf_ras_to_ijk(self):
        vol = self.volume

        test_point_ras = np.array([11.0, 24.0, 39.0], dtype=np.float64)
        expected_ijk = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        assert_array_almost_equal(vol.transf_ras_to_ijk(test_point_ras), expected_ijk)

        assert_array_almost_equal(
            vol.transf_ras_to_ijk(np.zeros(shape=(3,), dtype=np.float64)),
            np.array([-10.0, -10.0, -10.0], dtype=np.float64),
        )

        vol = Volume(ijk_to_ras=np.eye(4, dtype=np.float64))
        assert_array_almost_equal(
            vol.transf_ras_to_ijk(np.array([5.0, 5.0, 5.0], dtype=np.float64)),
            np.array([5.0, 5.0, 5.0], dtype=np.float64),
        )

    def test_04_Volume___call___(self):
        vol = self.volume

        vol(np.zeros(shape=(1, 3), dtype=np.float64))

    def test_05_Volume_get_line(self):
        vol = self.volume

        values, positions = vol.get_line(
            start=np.zeros(shape=(3,), dtype=np.float64),
            end=np.ones(shape=(3,), dtype=np.float64),
        )
        self.assertEqual(values.shape[0], positions.shape[0])

    def test_06_Volume_order(self):
        vol = self.volume

        with self.assertRaises(ValueError):
            vol.order = -1
