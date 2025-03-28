import unittest
from ransac_slicer.segment import Segment
from numpy.testing import assert_array_almost_equal
import numpy as np


class SegmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Default variables
        """
        cls.default_segment = Segment(
            start=np.zeros(shape=(3,), dtype=np.float64),
            end=np.array([0, 0, 2], dtype=np.float64),
        )
        cls.zero_segment = Segment()
        cls.unit_segment = Segment(
            start=np.zeros(shape=(3,), dtype=np.float64),
            end=np.array([0, 0, 1], dtype=np.float64),
        )

    def test_01_Segment_as_vector(self):
        assert_array_almost_equal(
            self.default_segment.as_vector(), np.array([0, 0, 2], dtype=np.float64)
        )
        assert_array_almost_equal(
            self.zero_segment.as_vector(), np.zeros(shape=(3,), dtype=np.float64)
        )
        assert_array_almost_equal(
            self.unit_segment.as_vector(), np.array([0, 0, 1], dtype=np.float64)
        )

    def test_02_Segment_distance_sqr(self):
        point = np.array([2, 0, 0], dtype=np.float64)
        self.assertAlmostEqual(self.default_segment.distance_sqr(point), 4.0)
        self.assertAlmostEqual(self.zero_segment.distance_sqr(point), 4.0)
        self.assertAlmostEqual(self.unit_segment.distance_sqr(point), 4.0)
