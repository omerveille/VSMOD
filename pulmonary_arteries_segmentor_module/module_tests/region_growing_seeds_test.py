import json
import unittest
from pathlib import Path

import networkx as nx
import numpy as np
import slicer
from networkx.readwrite import json_graph
from numpy.testing import assert_array_almost_equal
from ransac_slicer.graph_branches import restore_lists_from_graph
from ransac_slicer.region_growing_seeds import _compute_draw_order, paint_segments


class RegionGrowingSeedsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        slicer.mrmlScene.Clear()
        # get the resources path used of the test
        ressource_test_dir = (
            Path(__file__)
            .parent.parent.joinpath("Resources")
            .joinpath("Test")
            .absolute()
        )
        volume_path = ressource_test_dir.joinpath("IXI002-Guys-0828-MRA.nii.gz")
        branch_tree_path = ressource_test_dir.joinpath("graph_tree.json")
        cls.segmentation_not_merged_path = ressource_test_dir.joinpath(
            "segmentation_test_not_merged.npy"
        )
        cls.segmentation_merged_path = ressource_test_dir.joinpath(
            "segmentation_test_merged.npy"
        )

        with open(branch_tree_path) as f:
            js_graph = json.load(f)
        graph: nx.DiGraph = json_graph.node_link_graph(js_graph)

        cls.volume_node = slicer.util.loadVolume(str(volume_path))

        segmentation_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSegmentationNode"
        )
        segmentation_node.CreateDefaultDisplayNodes()
        segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(
            cls.volume_node
        )
        cls.segmentation_node = segmentation_node

        # Used to get the labelmap as a numpy array to compare
        cls.labelmap_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode"
        )

        (
            _,
            cls.names,
            cls.centerlines,
            _,
            cls.centerline_radius,
            edges,
            nodes,
            _,
            _,
        ) = restore_lists_from_graph(graph)
        cls.draw_order = _compute_draw_order(nodes, edges)

    def test_01_paint_segments_end_to_end(self):
        paint_segments(
            volume_node=self.volume_node,
            centerlines=self.centerlines,
            centerline_names=self.names,
            radius=self.centerline_radius,
            branch_draw_order=self.draw_order,
            segmentation_node=self.segmentation_node,
            reduction_factor=0.75,
            reduction_threshold=5.0,
            contour_distance=4,
            merge_all_vessels=False,
        )

        slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(
            self.segmentation_node, self.labelmap_node
        )
        labelmap_array: np.ndarray = slicer.util.arrayFromVolume(self.labelmap_node)
        true_segmentation = np.load(self.segmentation_not_merged_path)

        assert_array_almost_equal(true_segmentation, labelmap_array)
        del labelmap_array

        paint_segments(
            volume_node=self.volume_node,
            centerlines=self.centerlines,
            centerline_names=self.names,
            radius=self.centerline_radius,
            branch_draw_order=self.draw_order,
            segmentation_node=self.segmentation_node,
            reduction_factor=0.75,
            reduction_threshold=5.0,
            contour_distance=4,
            merge_all_vessels=True,
        )

        slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(
            self.segmentation_node, self.labelmap_node
        )
        labelmap_array: np.ndarray = slicer.util.arrayFromVolume(self.labelmap_node)
        true_segmentation = np.load(self.segmentation_merged_path)

        assert_array_almost_equal(true_segmentation, labelmap_array)

    def tearDown(self) -> None:
        slicer.mrmlScene.Clear()
