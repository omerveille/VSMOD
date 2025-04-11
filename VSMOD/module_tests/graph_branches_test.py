import json
from pathlib import Path
from datetime import datetime
import unittest
from .test_utils import get_resources_path, custom_iterable_assertion

import networkx as nx
from networkx.readwrite import json_graph
import numpy as np
from numpy.testing import assert_almost_equal

from ransac_slicer.graph_branches import (
    restore_lists_from_graph,
    GraphBranches,
)
from ransac_slicer.branch_tree import BranchTree, TreeColumnRole
from ransac_slicer.color_palettes import centerline_color, contour_points_color
from ransac_slicer.cylinder import Cylinder

import slicer
import qt


class Graph_branchesTest(unittest.TestCase):
    def setUp(self):
        # Define default tree widget and graph_branch for tests
        self.tree_widget = BranchTree()
        self.centerline_button = qt.QPushButton("Hide centerlines")
        self.contour_point_button = qt.QPushButton("Hide contours")
        self.lock_button = qt.QPushButton("Lock")
        self.lock_button.checked = False

        self.graph_branches = GraphBranches(
            tree_widget=self.tree_widget,
            centerline_button=self.centerline_button,
            contour_point_button=self.contour_point_button,
            lock_button=self.lock_button,
        )

    def tearDown(self):
        slicer.mrmlScene.Clear()

    def test_01_GraphBranches___init__(self):
        self.assertEqual(self.graph_branches.branch_list, [])
        self.assertEqual(self.graph_branches.nodes, [])
        self.assertEqual(self.graph_branches.edges, [])
        self.assertEqual(self.graph_branches.names, [])
        self.assertEqual(self.graph_branches.centerline_markups, [])
        self.assertEqual(self.graph_branches.contour_points_markups, [])
        self.assertAlmostEqual(self.graph_branches.centerline_text_size, 3.0)
        self.assertIs(self.graph_branches.tree_widget, self.tree_widget)

    def test_02_GraphBranches_create_new_markups(self):
        nb_cyls = 10
        self.graph_branches.branch_list.append(
            [
                Cylinder(radius=i, contour_points=np.random.random(3 * i).reshape(i, 3))
                for i in range(1, 1 + nb_cyls)
            ]
        )
        self.graph_branches.create_new_markup("b1")
        self.graph_branches.update_markup(0)

        self.assertEqual(len(self.graph_branches.centerline_markups), 1)
        self.assertEqual(len(self.graph_branches.contour_points_markups), 1)
        center_markup = self.graph_branches.centerline_markups[-1]
        contour_markup = self.graph_branches.contour_points_markups[-1]

        self.assertEqual(center_markup.GetName(), "b1_centers")
        self.assertEqual(contour_markup.GetName(), "b1_contours")

        self.assertAlmostEqual(
            center_markup.GetDisplayNode().GetTextScale(),
            self.graph_branches.centerline_text_size,
        )
        self.assertTupleEqual(
            tuple(center_markup.GetDisplayNode().GetSelectedColor()),
            tuple(centerline_color),
        )
        self.assertTupleEqual(
            tuple(contour_markup.GetDisplayNode().GetSelectedColor()),
            tuple(contour_points_color),
        )

        self.assertEqual(center_markup.GetNumberOfControlPoints(), nb_cyls)
        nb_contours_points = sum([(i + 1) for i in range(nb_cyls)])
        self.assertEqual(contour_markup.GetNumberOfControlPoints(), nb_contours_points)

    def test_03_GraphBranches_create_new_branch(self):
        nb_cyls = 10
        edge = (0, 1)
        branch = [
            Cylinder(radius=i, contour_points=np.random.random(3 * i).reshape(i, 3))
            for i in range(1, nb_cyls + 1)
        ]
        parent_node = None
        isFromSplitBranch = False
        self.graph_branches.create_new_branch(
            edge,
            branch,
            parent_node,
            isFromSplitBranch,
        )

        center_markup = self.graph_branches.centerline_markups[-1]
        contour_markup = self.graph_branches.contour_points_markups[-1]

        self.assertEqual(center_markup.GetName(), "b1_centers")
        self.assertEqual(contour_markup.GetName(), "b1_contours")

        self.assertAlmostEqual(
            center_markup.GetDisplayNode().GetTextScale(),
            self.graph_branches.centerline_text_size,
        )
        self.assertTupleEqual(
            tuple(center_markup.GetDisplayNode().GetSelectedColor()),
            tuple(centerline_color),
        )
        self.assertTupleEqual(
            tuple(contour_markup.GetDisplayNode().GetSelectedColor()),
            tuple(contour_points_color),
        )

        self.assertEqual(center_markup.GetNumberOfControlPoints(), nb_cyls)
        nb_contours_points = sum([(i + 1) for i in range(nb_cyls)])
        self.assertEqual(contour_markup.GetNumberOfControlPoints(), nb_contours_points)

        new_name = self.graph_branches.names[-1]
        custom_iterable_assertion(self.graph_branches.branch_list[-1], branch)
        self.assertIsNotNone(self.tree_widget.getTreeWidgetItem(new_name))

    def test_04_GraphBranches_truncate_branch(self):
        number_of_points = 10
        self.graph_branches.branch_list = [
            [Cylinder(contour_points=np.zeros(shape=(1, 3), dtype=np.float64))]
            * number_of_points
        ]
        self.graph_branches.centerline_markups = [
            slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode")
        ]
        self.graph_branches.contour_points_markups = [
            slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        ]
        self.graph_branches.truncate_branch_end(0, 2)

        self.assertEqual(len(self.graph_branches.branch_list[0]), 2)
        self.assertEqual(
            self.graph_branches.centerline_markups[0].GetNumberOfControlPoints(), 2
        )
        self.assertEqual(
            self.graph_branches.contour_points_markups[0].GetNumberOfControlPoints(), 2
        )

        self.graph_branches.branch_list = [
            [Cylinder(contour_points=np.zeros(shape=(1, 3), dtype=np.float64))]
            * number_of_points
        ]
        self.graph_branches.centerline_markups = [
            slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode")
        ]
        self.graph_branches.contour_points_markups = [
            slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        ]
        self.graph_branches.truncate_branch_begin(0, 8)

        self.assertEqual(len(self.graph_branches.branch_list[0]), 2)
        self.assertEqual(
            self.graph_branches.centerline_markups[0].GetNumberOfControlPoints(), 2
        )
        self.assertEqual(
            self.graph_branches.contour_points_markups[0].GetNumberOfControlPoints(), 2
        )

    def test_05_GraphBranches_update_visibility_button(self):
        for expected_text, visibility in zip(("Hide", "Show"), (True, False)):
            center_markup1 = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsCurveNode"
            )
            center_markup2 = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsCurveNode"
            )
            center_markup1.GetDisplayNode().SetVisibility(visibility)
            center_markup2.GetDisplayNode().SetVisibility(visibility)
            self.graph_branches.centerline_markups = [center_markup1, center_markup2]
            self.graph_branches.contour_points_markups = [
                center_markup1,
                center_markup2,
            ]
            self.graph_branches.update_visibility_button(
                TreeColumnRole.VISIBILITY_CENTER
            )

            self.assertIn(expected_text, self.centerline_button.text)

    def test_06_GraphBranches_split_branch(self):
        nb_cyls = 10
        edge = (0, 1)
        branch = [
            Cylinder(radius=i, contour_points=np.random.random(3 * i).reshape(i, 3))
            for i in range(1, nb_cyls + 1)
        ]
        parent_node = None
        isFromSplitBranch = False

        expected_cyl = branch[1]
        self.graph_branches.create_new_branch(
            edge,
            branch,
            parent_node,
            isFromSplitBranch,
        )
        cyl = self.graph_branches.split_branch(0, 1, None)

        self.assertEqual(len(self.graph_branches.branch_list), 2)
        self.assertEqual(len(self.graph_branches.names), 2)
        new_branch_name = self.graph_branches.names[-1]
        self.assertIsNotNone(self.tree_widget.getTreeWidgetItem(new_branch_name))
        self.assertEqual(cyl, expected_cyl)

    def test_07_GraphBranches_save_networkX(self):
        tempfile_path = Path(slicer.app.temporaryPath).joinpath(
            f"graph_tree_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
        )
        save_path = self.graph_branches.save_networkX(
            forced_path=tempfile_path, show_success_window=False
        )

        self.assertIsNotNone(save_path)
        self.addCleanup(self.remove_files, save_path)
        self.assertTrue(save_path.exists())

    def remove_files(self, file: Path):
        file.unlink(missing_ok=True)

    def test_08_GraphBranches_clear_all(self):
        self.graph_branches.branch_list = [[Cylinder()]]
        self.graph_branches.nodes = [[0] * 3]
        self.graph_branches.edges = [(0, 1)]
        self.graph_branches.names = ["b1"]
        self.graph_branches.centerline_markups = [
            slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode")
        ]
        self.graph_branches.contour_points_markups = [
            slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        ]
        self.tree_widget.insertAfterNode("b1", None)
        result = self.graph_branches.clear_all(auto_confirm=True)

        self.assertTrue(result)
        self.assertEqual(self.graph_branches.branch_list, [])
        self.assertEqual(self.graph_branches.nodes, [])
        self.assertEqual(self.graph_branches.edges, [])
        self.assertEqual(self.graph_branches.names, [])
        self.assertEqual(self.tree_widget._branchDict, {})

    def test_09_GraphBranches_on_stop_interaction(self):
        class DummyItem:
            def __init__(self):
                self.updated = False

            def updateText(self):
                self.updated = True

        dummy_item = DummyItem()
        self.graph_branches.current_tree_item = dummy_item
        self.graph_branches.on_stop_interaction()

        self.assertTrue(dummy_item.updated)

    def test_10_GraphBranches_on_item_clicked(self):
        self.tree_widget.insertAfterNode("b1", None)
        tree_item = self.tree_widget.getTreeWidgetItem("b1")
        self.graph_branches.names = ["b1"]

        center_markup = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode")
        center_markup.GetDisplayNode().SetVisibility(True)
        self.graph_branches.centerline_markups = [center_markup]

        contour_markup = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsFiducialNode"
        )
        contour_markup.GetDisplayNode().SetVisibility(True)
        self.graph_branches.contour_points_markups = [contour_markup]

        self.graph_branches.on_item_clicked(tree_item, TreeColumnRole.VISIBILITY_CENTER)
        self.assertFalse(center_markup.GetDisplayNode().GetVisibility())

        self.graph_branches.on_item_clicked(
            tree_item, TreeColumnRole.VISIBILITY_CONTOUR
        )
        self.assertFalse(contour_markup.GetDisplayNode().GetVisibility())

    def test_11_GraphBranches_on_item_renamed(self):
        self.graph_branches.names = ["b1"]
        center_markup = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode")
        contour_markup = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsFiducialNode"
        )
        self.graph_branches.centerline_markups = [center_markup]
        self.graph_branches.contour_points_markups = [contour_markup]
        self.graph_branches.on_item_renamed("b1", "b1_new")

        self.assertEqual(self.graph_branches.names[0], "b1_new")
        self.assertEqual(center_markup.GetName(), "b1_new_centers")
        self.assertEqual(contour_markup.GetName(), "b1_new_contours")

    def test_12_GraphBranches_on_key_pressed(self):
        self.tree_widget.insertAfterNode("b1", None)
        tree_item = self.tree_widget.getTreeWidgetItem("b1")
        flag = {"deleted": False}

        def dummy_delete(item):
            flag["deleted"] = True

        self.graph_branches.on_delete_item = dummy_delete
        self.graph_branches.on_key_pressed(tree_item, qt.Qt.Key_Delete)

        self.assertTrue(flag["deleted"])

    def test_13_GraphBranches_on_header_clicked(self):
        center_markup1 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode")
        center_markup2 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode")
        center_markup1.GetDisplayNode().SetVisibility(False)
        center_markup2.GetDisplayNode().SetVisibility(False)
        self.graph_branches.centerline_markups = [center_markup1, center_markup2]
        self.graph_branches.contour_points_markups = [center_markup1, center_markup2]
        self.graph_branches.on_header_clicked(TreeColumnRole.VISIBILITY_CENTER)

        self.assertTrue(
            all(
                m.GetDisplayNode().GetVisibility()
                for m in self.graph_branches.centerline_markups
            )
        )

    def test_14_GraphBranches_on_node_clicked(self):
        center_markup = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode")
        center_markup.SetName("b1_centers")
        self.graph_branches.names = ["b1"]
        self.tree_widget.insertAfterNode("b1", None)
        tree_item = self.tree_widget.getTreeWidgetItem("b1")

        self.graph_branches.node_selected = (-1, -1)
        self.assertTupleEqual(self.graph_branches.node_selected, (-1, -1))
        displayNode = center_markup.GetDisplayNode()
        displayNode.GetActiveComponentType = (
            lambda: slicer.vtkMRMLMarkupsDisplayNode.ComponentControlPoint
        )
        displayNode.GetActiveComponentIndex = lambda: 0
        self.graph_branches.on_node_clicked(center_markup, None)

        self.assertEqual(self.graph_branches.node_selected, (0, 0))
        self.assertIs(self.tree_widget.currentItem(), tree_item)

    def test_15_GraphBranches_on_remove(self):
        nb_cyls = 10
        branch = [
            Cylinder(
                center=[i] * 3,
                radius=i,
                contour_points=np.random.random(3 * i).reshape(i, 3),
            )
            for i in range(1, 1 + nb_cyls)
        ]

        self.graph_branches.branch_list.append(branch)
        self.graph_branches.names = ["b1"]
        self.graph_branches.edges = [(0, 1)]
        self.graph_branches.nodes = [branch[0].center, branch[-1].center]
        self.tree_widget.insertAfterNode("b1", None)
        tree_item = self.tree_widget.getTreeWidgetItem("b1")
        self.graph_branches.current_tree_item = tree_item
        self.graph_branches.node_selected = (0, 3)
        self.graph_branches.centerline_markups = [
            slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode")
        ]
        self.graph_branches.contour_points_markups = [
            slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        ]

        self.graph_branches.on_remove_end(tree_item)

        custom_iterable_assertion(branch[:4], self.graph_branches.branch_list[0])

        self.graph_branches.node_selected = (0, 1)
        self.graph_branches.on_remove_begin(tree_item)

        custom_iterable_assertion(branch[1:4], self.graph_branches.branch_list[0])
        assert_almost_equal(self.graph_branches.nodes[0], branch[1].center)
        assert_almost_equal(self.graph_branches.nodes[1], branch[3].center)

    def test_16_GraphBranches_delete_node(self):
        self.graph_branches.nodes = [[0] * 3, [1] * 3, [2] * 3]
        self.graph_branches.edges = [(0, 1), (1, 2)]
        self.graph_branches.delete_node(1)

        self.assertTupleEqual(tuple(self.graph_branches.nodes[0]), (0, 0, 0))
        self.assertTupleEqual(tuple(self.graph_branches.nodes[1]), (2, 2, 2))
        self.assertEqual(len(self.graph_branches.nodes), 2)

        self.assertTupleEqual(self.graph_branches.edges[0], (0, 1))
        self.assertTupleEqual(self.graph_branches.edges[1], (1, 1))
        self.assertEqual(len(self.graph_branches.edges), 2)

    def test_17_GraphBranches_on_delete_item(self):
        self.graph_branches.names = ["b1", "b2"]
        self.graph_branches.branch_list = [[], []]
        self.graph_branches.nodes = [[0] * 3, [1] * 3, [2] * 3]
        self.graph_branches.centerline_markups = [
            slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode")
            for _ in range(2)
        ]
        self.graph_branches.contour_points_markups = [
            slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
            for _ in range(2)
        ]
        self.graph_branches.edges = [(0, 1), (1, 2)]
        self.graph_branches.centerlines = [
            np.zeros(shape=(2, 3)),
            np.zeros(shape=(2, 3)),
        ]
        self.graph_branches.contours_points = [
            [[[0] * 3 for _ in range(3)] for _ in range(2)],
            [[[0] * 3 for _ in range(3)] for _ in range(2)],
        ]
        self.graph_branches.centerline_radius = [[0.5], [0.6]]
        self.tree_widget.insertAfterNode("b1", None)
        self.tree_widget.insertAfterNode("b2", "b1")
        self.graph_branches.on_delete_item(self.tree_widget.getTreeWidgetItem("b2"))

        self.assertNotIn("b2", self.graph_branches.names)
        self.assertIsNone(self.tree_widget.getTreeWidgetItem("b2"))

    def test_18_GraphBranches_on_merge_only_child(self):
        self.graph_branches.names = ["b1", "b2"]
        self.graph_branches.nodes = [[0] * 3, [1] * 3, [2] * 3]
        self.graph_branches.branch_list = [
            [
                Cylinder(radius=0.5, contour_points=np.zeros(shape=(1, 3))),
                Cylinder(radius=0.6, contour_points=np.zeros(shape=(1, 3))),
            ],
            [
                Cylinder(radius=0.7, contour_points=np.zeros(shape=(1, 3))),
                Cylinder(radius=0.8, contour_points=np.zeros(shape=(1, 3))),
            ],
        ]
        self.graph_branches.edges = [(0, 1), (1, 2)]
        self.graph_branches.centerline_markups = [
            slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode")
            for _ in range(2)
        ]
        self.graph_branches.contour_points_markups = [
            slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
            for _ in range(2)
        ]

        self.tree_widget.insertAfterNode("b1", None)
        self.tree_widget.insertAfterNode("b2", "b1")
        self.graph_branches.on_merge_only_child("b1")

        expected_branch = [
            [
                Cylinder(radius=0.5, contour_points=np.zeros(shape=(1, 3))),
                Cylinder(radius=0.6, contour_points=np.zeros(shape=(1, 3))),
                Cylinder(radius=0.8, contour_points=np.zeros(shape=(1, 3))),
            ]
        ]
        custom_iterable_assertion(expected_branch, self.graph_branches.branch_list)

    def test_19_GraphBranches_extend_root_from_begin(self):
        input_branch = [
            Cylinder(contour_points=np.zeros(shape=(1, 3), dtype=np.float64))
            for _ in range(10)
        ]
        self.graph_branches.branch_list.append(input_branch)
        self.graph_branches.edges = [(0, 1)]
        self.graph_branches.nodes = [
            np.zeros(shape=(3,), dtype=np.float64),
            np.zeros(shape=(3,), dtype=np.float64),
        ]
        self.graph_branches.centerline_markups = [
            slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode")
        ]
        self.graph_branches.contour_points_markups = [
            slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        ]

        branch = [
            Cylinder(contour_points=np.zeros(shape=(i, 3), dtype=np.float64))
            for i in range(10)
        ]
        root_idx = 0
        self.graph_branches.extend_root_from_begin(
            branch=branch,
            root_idx=root_idx,
        )

        expected_branch = branch[::-1][:-1] + input_branch

        self.assertEqual(len(self.graph_branches.branch_list), 1)
        custom_iterable_assertion(expected_branch, self.graph_branches.branch_list[0])

    def test_20_restore_lists_from_graph(self):
        graph_path = get_resources_path().joinpath("graph_tree.json")
        with open(graph_path) as f:
            js_graph = json.load(f)
        graph: nx.DiGraph = json_graph.node_link_graph(js_graph)
        self.graph_branches.load_branches_from_graph(graph)

        lists = restore_lists_from_graph(graph)

        tempfile_path = Path(slicer.app.temporaryPath).joinpath(
            f"graph_tree_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
        )
        save_path = self.graph_branches.save_networkX(
            forced_path=tempfile_path, show_success_window=False
        )
        self.assertIsNotNone(save_path)
        self.addCleanup(self.remove_files, save_path)
        self.assertTrue(save_path.exists())

        with open(save_path) as f:
            js_graph = json.load(f)
        graph: nx.DiGraph = json_graph.node_link_graph(js_graph)
        new_lists = restore_lists_from_graph(graph)

        for item_1, item_2 in zip(lists, new_lists):
            if isinstance(item_1, list):
                self.assertEqual(len(item_1), len(item_2))
                custom_iterable_assertion(item_1, item_2)
            else:
                self.assertDictEqual(item_1, item_2)
