import unittest

import numpy as np
import qt
import slicer
from ransac_slicer.branch_tree import BranchTree
from ransac_slicer.cylinder_ransac import (
    Config,
)
from ransac_slicer.graph_branches import GraphBranches
from ransac_slicer.popup_utils import CustomStatusDialog
from ransac_slicer.ransac import interpolate_centerline, interpolate_point, run_ransac
from ransac_slicer.volume import Volume

from .test_utils import get_resources_path, load_object_from_path


class RansacTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        slicer.mrmlScene.Clear()

        ressource_test_dir = get_resources_path()
        volume_path = ressource_test_dir.joinpath("IXI002-Guys-0828-MRA.nii.gz")

        # Prepare the volume object
        volume = load_object_from_path(volume_path)
        cls.vol = Volume.from_scalar_volume(volume)

        # Prepare the ransac config object
        cls.cfg = Config(
            percent_inliers=0.75,
            threshold=0.3,
            angle_max=np.pi / 2,
            nb_test_min=20000,
            nb_test_max=20000,
            nb_iter=10,
        )

        cls.resources_path = get_resources_path().joinpath("ransac")

    def test_01_interpolate_point(self):
        cyl_0 = load_object_from_path(self.resources_path.joinpath("cyl_0.pickle"))
        cyl_1 = load_object_from_path(self.resources_path.joinpath("cyl_1.pickle"))

        centers, _ = interpolate_point(
            cyl_0=cyl_0, cyl_1=cyl_1, vol=self.vol, cfg=self.cfg, distance=0.1
        )
        self.assertGreaterEqual(len(centers), 2)

        centers, _ = interpolate_point(
            cyl_0=cyl_0, cyl_1=cyl_1, vol=self.vol, cfg=self.cfg, distance=1000
        )
        self.assertEqual(len(centers), 0)

    def test_02_interpolate_centerline(self):
        cylinders = load_object_from_path(
            self.resources_path.joinpath("cyl_list.pickle")
        )
        contour_points = load_object_from_path(
            self.resources_path.joinpath("contour_points_cylinders.pickle")
        )

        centerline, _, _ = interpolate_centerline(
            cylinders=cylinders,
            contour_points=contour_points,
            vol=self.vol,
            cfg=self.cfg,
            distance=0.4,
        )
        self.assertGreaterEqual(centerline.shape[0], len(cylinders))

        centerline, _, _ = interpolate_centerline(
            cylinders=cylinders,
            contour_points=contour_points,
            vol=self.vol,
            cfg=self.cfg,
            distance=1000,
        )
        self.assertEqual(centerline.shape[0], len(cylinders))

    def test_03_run_ransac(self):
        starting_point = load_object_from_path(
            self.resources_path.joinpath("starting_point.npy")
        )
        direction_point = load_object_from_path(
            self.resources_path.joinpath("direction_point.npy")
        )
        tree_widget = BranchTree()
        centerline_button = qt.QPushButton("Hide centerlines")
        contour_point_button = qt.QPushButton("Hide contours")
        lock_button = qt.QPushButton("Lock")
        lock_button.checked = False

        graph_branches = GraphBranches(
            tree_widget=tree_widget,
            centerline_button=centerline_button,
            contour_point_button=contour_point_button,
            lock_button=lock_button,
        )

        progress_dialog = CustomStatusDialog(
            windowTitle="Computing centerline...",
            text="Please wait",
            width=300,
            height=50,
        )

        # We basically assert nothing wrong happens
        run_ransac(
            vol=self.vol,
            starting_point=starting_point,
            direction_point=direction_point,
            starting_radius=0.7,
            percent_inlier_points=75,
            inlier_threshold=30,
            centerline_resolution=0.5,
            maximum_turn_angle=90,
            min_number_of_attempts=20000,
            max_number_of_attempts=20000,
            max_number_of_cylinders=1000,
            smart_diameter_selection=False,
            graph_branches=graph_branches,
            isNewBranch=False,
            progress_dialog=progress_dialog,
        )
