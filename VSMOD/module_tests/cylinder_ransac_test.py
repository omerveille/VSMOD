import unittest

import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
import json
import slicer

from ransac_slicer.cylinder_ransac import (
    Config,
    next_cylinder,
    sample,
    sample_around_cylinder,
    track_branch,
)
from ransac_slicer.volume import Volume
from ransac_slicer.cylinder import Cylinder
from ransac_slicer.graph_branches import restore_lists_from_graph
from .test_utils import get_resources_path, load_object_from_path
from ransac_slicer.popup_utils import CustomStatusDialog


class Cylinder_ransacTest(unittest.TestCase):
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

        # Config to ensure test fails
        cls.fail_cfg = Config(
            percent_inliers=1.0,
            threshold=0,
            angle_max=np.pi / 2,
            nb_test_min=0,
            nb_test_max=0,
            nb_iter=0,
        )

        graph_path = get_resources_path().joinpath("graph_tree.json")
        with open(graph_path) as f:
            js_graph = json.load(f)
        graph: nx.DiGraph = json_graph.node_link_graph(js_graph)
        branch_list, _, _, _, _ = restore_lists_from_graph(graph)
        cls.example_cylinder = branch_list[0][0]

    def test_01_sample(self):
        points = sample(
            vol=self.vol,
            center=np.array(self.vol._vol.shape, dtype=np.float64) // 2,
            radius=2.0,
            n_samples=self.cfg.n_samples,
            dirs=self.cfg.ray_dir_set,
        )
        self.assertTupleEqual(points.shape, self.cfg.ray_dir_set.shape)

    def test_02_sample_around_cylinder(self):
        self.assertIsNotNone(
            sample_around_cylinder(
                vol=self.vol, cyl=self.example_cylinder, cfg=self.cfg
            )
        )
        self.assertIsNone(
            sample_around_cylinder(vol=self.vol, cyl=Cylinder(), cfg=self.fail_cfg)
        )

    def test_03_next_cylinder(self):
        self.assertIsNotNone(next_cylinder(self.vol, self.example_cylinder, self.cfg))
        self.assertIsNone(next_cylinder(self.vol, Cylinder(), self.fail_cfg))

    def test_04_track_branch(self):
        progress_dialog = CustomStatusDialog(
            windowTitle="Computing centerline...",
            text="Please wait",
            width=300,
            height=50,
        )

        tracked_cylinders = track_branch(
            vol=self.vol,
            cyl=self.example_cylinder,
            cfg=self.cfg,
            already_tracked_cylinders=[],
            progress_dialog=progress_dialog,
        )
        self.assertTrue(len(tracked_cylinders) > 1)

        tracked_cylinders = track_branch(
            vol=self.vol,
            cyl=self.example_cylinder,
            cfg=self.fail_cfg,
            already_tracked_cylinders=[],
            progress_dialog=progress_dialog,
        )
        self.assertTrue(len(tracked_cylinders) == 0)
