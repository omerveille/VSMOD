import json
import unittest

import networkx as nx
import numpy as np
import slicer
from networkx.readwrite import json_graph
from numpy.testing import assert_array_almost_equal
from ransac_slicer.graph_branches import restore_lists_from_graph
from ransac_slicer.region_growing_seeds import paint_segments
from .test_utils import get_resources_path


class RegionGrowingSeedsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        slicer.mrmlScene.Clear()

        ressource_test_dir = get_resources_path()
        volume_path = ressource_test_dir.joinpath("IXI002-Guys-0828-MRA.nii.gz")
        branch_tree_path = ressource_test_dir.joinpath("graph_tree.json")

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

        (
            cls.branch_list,
            cls.names,
            cls.edges,
            cls.nodes,
            cls.edge_name_table,
        ) = restore_lists_from_graph(graph)

    def test_01_paint_segments_end_to_end(self):
        labelmap_array = paint_segments(
            volume_node=self.volume_node,
            branches=self.branch_list,
            centerline_names=self.names,
            nodes=self.nodes,
            edges=self.edges,
            segmentation_node=self.segmentation_node,
            reduction_factor=0.75,
            radius_reduction_threshold=5.0,
            contour_distance=4,
            merge_all_vessels=False,
        )

        assert_array_almost_equal(
            np.arange(stop=len(self.branch_list) + 2), np.unique(labelmap_array)
        )
        del labelmap_array

        labelmap_array = paint_segments(
            volume_node=self.volume_node,
            branches=self.branch_list,
            centerline_names=self.names,
            nodes=self.nodes,
            edges=self.edges,
            segmentation_node=self.segmentation_node,
            reduction_factor=0.75,
            radius_reduction_threshold=5.0,
            contour_distance=4,
            merge_all_vessels=True,
        )

        assert_array_almost_equal(np.arange(stop=3), np.unique(labelmap_array))

    def tearDown(self) -> None:
        slicer.mrmlScene.Clear()
