from typing import Union
import networkx as nx
from networkx.readwrite import json_graph
import json
import numpy as np
from .popup_utils import CustomProgressBar
from datetime import datetime
import slicer
import qt
from pathlib import Path
from .cylinder import Cylinder
from .branch_tree import BranchTree, TreeColumnRole, Icons
from .color_palettes import centerline_color, contour_points_color


def restore_lists_from_graph(graph: nx.DiGraph):
    """
    Extract all the necessary lists to load a saved tree architecture into a GraphBranches object.

    Parameters
        ----------

        graph: nx.DiGraph
            A DiGraph loaded from a file, which represent a saved tree architecture.

        Returns
        ----------

        All the lists necessary to regenerate the tree architecture in the UI.
    """
    branch_list = []
    names = []
    edges = []
    nodes = []
    edge_name_table = {0: None}

    for a, b in CustomProgressBar(
        iterable=graph.edges,
        quantity_to_measure="branch loaded",
        windowTitle="Restoring tree architecture...",
        width=300,
    ):
        # Restoring lists
        names.append(graph[a][b]["name"])
        edges.append((a, b))
        edge_name_table[b] = graph[a][b]["name"]

        centers = graph[a][b]["centerline"]
        # (THE SECOND PART IS TO REMOVE LATER ITS AN ARTIFACT FROM THE PAST)
        radius = graph[a][b].get("radius", None)
        # (THE SECOND PART IS TO REMOVE LATER ITS AN ARTIFACT FROM THE PAST)
        directions = graph[a][b].get(
            "direction",
            [np.array([0, 0, 1], dtype=np.float64) for _ in range(len(centers))],
        )
        # (THE SECOND PART IS TO REMOVE LATER ITS AN ARTIFACT FROM THE PAST)
        heights = graph[a][b].get("height", [-1 for _ in range(len(centers))])
        contour_points = graph[a][b]["contour_points"]

        # (TO REMOVE ITS AN ARTIFACT FROM THE PAST)
        # Recompute radius if not existant
        if radius is None:
            radius = [
                np.linalg.norm(
                    np.array(graph[a][b]["contour_points"][k])
                    - np.array(graph[a][b]["centerline"][k]),
                    axis=1,
                ).min()
                for k in range(len(graph[a][b]["centerline"]))
            ]

        branch_list.append(
            [
                Cylinder(center=c, radius=r, direction=d, height=h, contour_points=cp)
                for c, r, d, h, cp in zip(
                    centers, radius, directions, heights, contour_points
                )
            ]
        )

    for node in graph.nodes(data=True):
        nodes.append(node[1]["pos"])

    return (
        branch_list,
        names,
        edges,
        nodes,
        edge_name_table,
    )


class GraphBranches:
    """
    Class which hold the graph of all the vessels segmented.

    The edges of the graph hold the points, and the node denotes bifurcation.

    Note:
    A branch can split in an infinite amount of childs.
    """

    def __init__(
        self,
        tree_widget: BranchTree,
        centerline_button: qt.QPushButton,
        contour_point_button: qt.QPushButton,
        lock_button: qt.QPushButton,
    ) -> None:
        self.branch_list: list[
            list[Cylinder]
        ] = []  # list of shape (n,m) with n = number of branches and m = number of cylinder in the current branch
        self.nodes: list[
            np.ndarray
        ] = []  # list of nodes which are the birfucation + root + leaves
        self.edges: list[tuple[int, int]] = []  # list of tuple for edges between nodes
        self.names: list[str] = []  # list of names in each edges

        self.centerline_markups: list[
            slicer.vtkMRMLMarkupsCurveNode
        ] = []  # list of markups for centers line
        self.contour_points_markups: list[
            slicer.vtkMRMLMarkupsFiducialNode
        ] = []  # list of markups for contour points

        self.tree_widget: BranchTree = tree_widget
        self.centerline_button: qt.QPushButton = centerline_button
        self.contour_point_button: qt.QPushButton = contour_point_button
        self.lock_button: qt.QPushButton = lock_button
        self.centerline_text_size: float = 3.0

        self.current_tree_item: qt.QTreeWidgetItem = None
        self.tree_widget.connect(
            "itemClicked(QTreeWidgetItem *, int)", self.on_item_clicked
        )
        self.tree_widget.itemRenamed.connect(self.on_item_renamed)
        self.tree_widget.itemRemoveBegin.connect(self.on_remove_begin)
        self.tree_widget.itemRemoveEnd.connect(self.on_remove_end)
        self.tree_widget.itemDeleted.connect(self.on_delete_item)
        self.tree_widget.keyPressed.connect(self.on_key_pressed)
        self.tree_widget.headerClicked.connect(self.on_header_clicked)

        self.node_selected = (-1, -1)

    def update_markup(self, branch_idx: int):
        """
        Update the markup in the scene by reloading its points.
        Used after an update on branch's cylinder.

        Parameters
        ----------

        branch_idx: the index of the branch that will be updated.

        """
        slicer.util.updateMarkupsControlPointsFromArray(
            self.centerline_markups[branch_idx],
            np.array(
                [cyl.center for cyl in self.branch_list[branch_idx]], dtype=np.float64
            ),
        )
        slicer.util.updateMarkupsControlPointsFromArray(
            self.contour_points_markups[branch_idx],
            np.vstack([cyl.contour_points for cyl in self.branch_list[branch_idx]]),
        )

    def create_new_markup(self, name: str):
        """
        Create a new markup for the centerline and the associated contour points.

        Parameters
        ----------

        name: name of the branch.
        """
        centerline_markup = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsCurveNode"
        )

        centerline_markup.SetName(name + "_centers")
        centerline_markup.GetDisplayNode().SetTextScale(self.centerline_text_size)
        centerline_markup.AddObserver(
            slicer.vtkMRMLMarkupsNode.PointStartInteractionEvent, self.on_node_clicked
        )
        centerline_markup.GetDisplayNode().SetSelectedColor(*centerline_color)

        contour_points_markup = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsFiducialNode"
        )

        contour_points_markup.GetDisplayNode().SetTextScale(0)
        contour_points_markup.GetDisplayNode().SetVisibility(False)
        contour_points_markup.GetDisplayNode().SetSelectedColor(*contour_points_color)
        contour_points_markup.SetName(name + "_contours")

        self.centerline_markups.append(centerline_markup)
        self.contour_points_markups.append(contour_points_markup)

        if self.lock_button.checked:
            centerline_markup.LockedOn()
            contour_points_markup.LockedOn()

        self.update_visibility_button(TreeColumnRole.VISIBILITY_CENTER)
        self.update_visibility_button(TreeColumnRole.VISIBILITY_CONTOUR)

    def create_new_branch(
        self,
        edge,
        branch: list[Cylinder],
        parent_node: Union[str, None] = None,
        isFromSplitBranch: bool = False,
    ):
        """
        Update the graph with the new edge and create associated markups for the centerline and the contour points.

        Parameters
        ----------

        edge: the edge to be added to the graph.
        branch: the cylinders that define the branch.
        parent_node: id of its parent node, None if it is the root.
        isFromSplitBranch: flag to check if this new branch is from a split, if it is not from a split we may
        merge branch with its single children.
        """
        self.branch_list.append(branch)

        self.edges.append(edge)
        new_name = "b" + str(len(self.edges))
        self.names.append(new_name)

        self.create_new_markup(new_name)
        self.update_markup(len(self.branch_list) - 1)

        self.tree_widget.insertAfterNode(
            nodeId=new_name,
            parentNodeId=parent_node,
            becomeIntermediaryParent=isFromSplitBranch,
        )

        if not isFromSplitBranch:
            self.on_merge_only_child(parent_node)

    def truncate_branch_begin(self, branch_idx: int, node_idx: int):
        """
        Truncate the beginning of a branch, deleting node before a certain index.

        Parameters
        ----------

        branch_idx: index of the branch updated.
        node_idx: index of the last point of the branch.
        """
        self.branch_list[branch_idx] = self.branch_list[branch_idx][node_idx:]
        self.update_markup(branch_idx)

    def truncate_branch_end(self, branch_idx: int, node_idx: int):
        """
        Truncate the end of a branch, deleting node after a certain index.

        Parameters
        ----------

        branch_idx: index of the branch updated.
        node_idx: index of the last point of the branch.
        """
        self.branch_list[branch_idx] = self.branch_list[branch_idx][:node_idx]
        self.update_markup(branch_idx)

    def update_visibility_button(self, column: TreeColumnRole):
        """
        Update the text of the button according to the action being the most annoying to do.
        For example, if two out of three items are visible, the action will be to turn them invisible.

        Parameters
        ----------

        column: flag to indicate the column to be updated.
        """
        markup_list, button = (
            (self.centerline_markups, self.centerline_button)
            if column == TreeColumnRole.VISIBILITY_CENTER
            else (self.contour_points_markups, self.contour_point_button)
        )
        majority_visibility = not (
            np.sum([markup.GetDisplayNode().GetVisibility() for markup in markup_list])
            >= max(1, (len(markup_list) // 2))
        )
        button.text = (
            button.text.replace("Hide", "Show")
            if majority_visibility
            else button.text.replace("Show", "Hide")
        )

    def split_branch(
        self, idx_branch: int, idx_cyl: int, parent_node: Union[str, None]
    ):
        """
        Split a branch into two parts.
        Triggered when the user adds a new branch, and the closest cylinder to that branch
        is located in the middle of a branch. Thus, this branch first part become parent of its second part
        and the newly branch created.

        Parameters
        ----------

        idx_branch: index of the branch splited.
        idx_cyl: index of the closest cylinder to the new branch created.
        parent_node: name of the parent of the branch splited, None if it is the root.

        Returns
        ----------

        The beginning parts of the newly branch created.

        """
        # Modify old branch which became a parent
        branch = self.branch_list[idx_branch]
        self.truncate_branch_end(idx_branch, idx_cyl + 1)

        # Update edges
        self.nodes.append(branch[idx_cyl].center)
        old_end = self.edges[idx_branch][1]
        self.edges[idx_branch] = (self.edges[idx_branch][0], len(self.nodes) - 1)

        # Create new branch from the old one but as a child
        self.create_new_branch(
            (len(self.nodes) - 1, old_end),
            branch[idx_cyl:],
            parent_node,
            True,
        )

        return branch[idx_cyl]

    def save_networkX(
        self,
        forced_path: Union[None, Path, str] = None,
        show_success_window: bool = True,
    ) -> Union[None, Path]:
        """
        Save the graph created as a networkx .JSON file if the user select a valid file.

        Parameters
        ----------
        forced_path : Union[None, Path]
            Path used to save the networkx graph, override the UI decision and does not make it appears if set.
            None by default.

        show_success_window : bool
          Wether to show a message in case of a successful save (disabled while testing).

        Returns
        -------

        Union[None, str]
          None if the user did not enter a folder path, otherwise returns the name of the saved file.
        """

        if forced_path is not None:
            file_save_path = forced_path
        else:
            dialog = qt.QFileDialog()
            file_save_path = dialog.getSaveFileName(
                None,
                "Save as",
                Path.home().joinpath(
                    f"graph_tree_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.json"
                ),
            )

        # cancel any action if the user cancel / close the window / press escape
        if not file_save_path:
            return

        file_save_path = Path(file_save_path)
        if not file_save_path.name.endswith(".json"):
            file_save_path = file_save_path.with_name(file_save_path.name + ".json")

        # Create graph Network X with node = bifurcation and edges = branches
        branch_graph = nx.DiGraph()

        for i, n in enumerate(self.nodes):
            branch_graph.add_node(i, pos=n)
        for i, e in enumerate(self.edges):
            centerline = []
            radius = []
            direction = []
            height = []
            contour_points = []

            for cyl in self.branch_list[i]:
                centerline.append(cyl.center)
                radius.append(cyl.radius)
                direction.append(cyl.direction)
                height.append(cyl.height)
                contour_points.append(cyl.contour_points)

            branch_graph.add_edge(
                e[0],
                e[1],
                name=self.names[i],
                centerline=centerline,
                radius=radius,
                direction=direction,
                height=height,
                contour_points=contour_points,
            )

        def ndarray_to_list(data):
            if isinstance(data, np.ndarray):
                return data.tolist()
            if isinstance(data, list):
                return [ndarray_to_list(item) for item in data]
            elif isinstance(data, dict):
                return {k: ndarray_to_list(v) for k, v in data.items()}
            else:
                return data

        # save to JSON
        data = json_graph.node_link_data(branch_graph)
        data_list = ndarray_to_list(data)
        with open(file_save_path, "w") as outfile:
            json.dump(data_list, outfile, indent=4)

        if show_success_window:
            slicer.util.infoDisplay(
                f"The graph has been successfully exported to :\n{file_save_path.parent}",
                windowTitle="Success",
            )
        return file_save_path

    def load_branches_from_graph(self, graph: nx.DiGraph):
        """
        Restore the branch contained in a networkx digraph file.
        Make sure the tree has been cleared BEFORE calling this function.

        Parameters
        ----------

        graph :
            Graph containing information that can be loading into an empty graph_branch tree.
        """
        with slicer.util.tryWithErrorDisplay(
            "Failed to restore tree architecture.", waitCursor=True
        ):
            (
                self.branch_list,
                self.names,
                self.edges,
                self.nodes,
                edge_name_table,
            ) = restore_lists_from_graph(graph)

            for idx, name in CustomProgressBar(
                iterable=list(enumerate(self.names)),
                quantity_to_measure="branch added",
                windowTitle="Restoring tree architecture...",
                width=300,
            ):
                self.create_new_markup(name=name)
                self.update_markup(idx)

            for a, b in nx.edge_dfs(graph):
                current_edge_name = graph[a][b]["name"]
                parent_edge_name = edge_name_table[a]
                self.tree_widget.insertAfterNode(
                    nodeId=current_edge_name, parentNodeId=parent_edge_name
                )

    def clear_all(self, auto_confirm: bool = False) -> bool:
        """
        Clear the whole graph after confirmation.

        Parameters
        ----------
        auto_confirm :
            If set to True, allows deletion without UI confirmation.
            False by default.

        Returns
        ----------

        True if the whole graph has been deleted else False.
        """

        if not auto_confirm:
            msg = qt.QMessageBox()
            msg.setIcon(qt.QMessageBox.Warning)
            msg.setWindowTitle("Confirmation")
            msg.setText("Are you sure you want to clear the tree ?")
            msg.setStandardButtons(qt.QMessageBox.Yes | qt.QMessageBox.No)

            if msg.exec_() != qt.QMessageBox.Yes:
                return False

        self.branch_list = []
        self.nodes = []
        self.edges = []
        self.names = []

        for _ in CustomProgressBar(
            iterable=range(len(self.centerline_markups)),
            quantity_to_measure="branch deleted",
            windowTitle="Clearing tree architecture...",
            width=300,
        ):
            slicer.mrmlScene.RemoveNode(self.centerline_markups.pop())
            slicer.mrmlScene.RemoveNode(self.contour_points_markups.pop())

        self.tree_widget.clear()
        self.update_visibility_button(TreeColumnRole.VISIBILITY_CENTER)
        self.update_visibility_button(TreeColumnRole.VISIBILITY_CONTOUR)
        return True

    def on_stop_interaction(self):
        if self.current_tree_item is not None:
            self.current_tree_item.updateText()

    def on_item_clicked(self, treeItem, column: TreeColumnRole):
        """
        On item clicked in the tree view do the associated action.

        Parameters
        ----------

        treeItem: tree item on which the user clicked a column.
        column: flag the indicate which column has been clicked.
        """
        self.current_tree_item = treeItem
        node_id = treeItem.nodeId
        branch_id = self.names.index(node_id)
        if column == TreeColumnRole.VISIBILITY_CENTER:
            is_visible = (
                self.centerline_markups[branch_id].GetDisplayNode().GetVisibility()
            )
            self.centerline_markups[branch_id].GetDisplayNode().SetVisibility(
                not is_visible
            )
            self.tree_widget._branchDict[node_id].setIcon(
                TreeColumnRole.VISIBILITY_CENTER,
                Icons.visibleOff if is_visible else Icons.visibleOn,
            )
            self.update_visibility_button(column)
        elif column == TreeColumnRole.VISIBILITY_CONTOUR:
            is_visible = (
                self.contour_points_markups[branch_id].GetDisplayNode().GetVisibility()
            )
            self.contour_points_markups[branch_id].GetDisplayNode().SetVisibility(
                not is_visible
            )
            self.tree_widget._branchDict[node_id].setIcon(
                TreeColumnRole.VISIBILITY_CONTOUR,
                Icons.visibleOff if is_visible else Icons.visibleOn,
            )
            self.update_visibility_button(column)
        elif column == TreeColumnRole.DELETE:
            self.on_delete_item(treeItem)
            self.update_visibility_button(TreeColumnRole.VISIBILITY_CENTER)
            self.update_visibility_button(TreeColumnRole.VISIBILITY_CONTOUR)

    def on_item_renamed(self, previous: str, new: str):
        """
        Rename the markup when the associated branch is renamed.

        Parameters
        ----------

        previous: previous name of the branch.
        new: new name of the branch.
        """
        branch_id = self.names.index(previous)
        self.names[branch_id] = new
        self.centerline_markups[branch_id].SetName(new + "_centers")
        self.contour_points_markups[branch_id].SetName(new + "_contours")

    def on_key_pressed(self, treeItem, key):
        """
        On delete key pressed, delete the current item if any selected.
        Can be modified to manage shortcuts.
        """
        if key == qt.Qt.Key_Delete:
            self.on_delete_item(treeItem)

    def on_header_clicked(self, column: TreeColumnRole):
        """
        On header clicked in the tree view do the associated action.

        Parameters
        ----------

        column: flag to indicate the column clicked.
        """

        def change_majority_visibility(markup_list, column, button):
            majority_visibility = not (
                np.sum(
                    [markup.GetDisplayNode().GetVisibility() for markup in markup_list]
                )
                >= max(1, (len(markup_list) // 2))
            )
            icon = Icons.visibleOn if majority_visibility else Icons.visibleOff
            for markup in markup_list:
                markup.GetDisplayNode().SetVisibility(majority_visibility)
            for branch in self.tree_widget._branchDict.values():
                branch.setIcon(column, icon)
            self.update_visibility_button(column)

        if column == TreeColumnRole.VISIBILITY_CENTER:
            change_majority_visibility(
                self.centerline_markups, column, self.centerline_button
            )
        elif column == TreeColumnRole.VISIBILITY_CONTOUR:
            change_majority_visibility(
                self.contour_points_markups, column, self.contour_point_button
            )

    def on_node_clicked(self, caller, event):
        """
        Callback function when the user click a node in the 3D slicer's 3D view.

        Parameters
        ----------

        caller: markup node the user clicked.
        """
        displayNode = caller.GetDisplayNode()
        if (
            displayNode.GetActiveComponentType()
            == slicer.vtkMRMLMarkupsDisplayNode.ComponentControlPoint
        ):
            node_id = displayNode.GetActiveComponentIndex()

            branch_name = "_".join(caller.GetName().split("_")[:-1])
            branch_id = self.names.index(branch_name)
            self.node_selected = (branch_id, node_id)

            tree_item = self.tree_widget.getTreeWidgetItem(branch_name)
            self.tree_widget.lastItemSelectInScene = tree_item
            self.tree_widget.scrollToItem(tree_item)
            self.tree_widget.setCurrentItem(tree_item)

    def on_remove_begin(self, treeItem):
        """
        Callback function when the user choose to delete the end of a branch.

        Parameters
        ----------

        treeItem: tree item in which the user wants to delete the end.
        """

        node_id = treeItem.nodeId
        branch_id = self.names.index(node_id)
        branch_selected, branch_node_id = self.node_selected

        if branch_selected == -1:
            msg = qt.QMessageBox()
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setWindowTitle("Error")
            msg.setText("No node selected.")
            msg.exec_()
            return

        if branch_id != branch_selected:
            msg = qt.QMessageBox()
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setWindowTitle("Error")
            msg.setText("The node selected do not belong to this branch.")
            msg.exec_()
            return

        if branch_node_id == 0:
            return

        self.truncate_branch_begin(branch_id, branch_node_id)
        edges_node_id = self.edges[branch_id][0]
        self.nodes[edges_node_id] = self.branch_list[branch_id][0].center

    def on_remove_end(self, treeItem):
        """
        Callback function when the user choose to delete the beginning of the root.

        Parameters
        ----------

        treeItem: tree item in which the user wants to delete the end.
        """
        node_id = treeItem.nodeId
        branch_id = self.names.index(node_id)
        branch_selected, branch_node_id = self.node_selected

        if branch_selected == -1:
            msg = qt.QMessageBox()
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setWindowTitle("Error")
            msg.setText("No node selected.")
            msg.exec_()
            return

        if branch_id != branch_selected:
            msg = qt.QMessageBox()
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setWindowTitle("Error")
            msg.setText("The node selected do not belong to this branch.")
            msg.exec_()
            return

        # Nothing to delete
        if branch_node_id == len(self.branch_list[branch_id]) - 1:
            return

        edges_node_id = self.edges[branch_id][1]
        self.nodes[edges_node_id] = self.branch_list[branch_id][branch_node_id].center
        self.truncate_branch_end(branch_id, branch_node_id + 1)

    def delete_node(self, index: int):
        """
        Delete a node from the graph

        Parameters
        ----------

        index: index of the node to be removed.
        """
        self.nodes.pop(index)
        for i in range(len(self.edges)):
            n1, n2 = self.edges[i]
            if n1 > index:
                n1 -= 1
            if n2 > index:
                n2 -= 1
            self.edges[i] = n1, n2

    def on_delete_item(self, treeItem, showPopupForNonLeaf=True):
        """
        Callback function when the user choose to delete a branch from the tree view.
        If the branch has childs, a confirmation is required.

        Parameters
        ----------

        treeItem: tree item the user wish to delete.
        showPopupForNonLeaf: flag to indicate that it is the initial call to the function, default is True
        so that it does not recursivly ask for deletion confirmation.
        """
        self.on_stop_interaction()
        node_id = treeItem.nodeId

        if self.tree_widget.isRoot(node_id):
            slicer.util.errorDisplay(
                text="You can't delete the root", windowTitle="Error"
            )
            return

        children = [
            self.tree_widget.getTreeWidgetItem(n_id)
            for n_id in self.tree_widget.getChildrenNodeId(node_id)
        ]

        if len(children) != 0 and showPopupForNonLeaf:
            msg = qt.QMessageBox()
            msg.setIcon(qt.QMessageBox.Warning)
            msg.setWindowTitle("Confirmation")
            msg.setText(
                f"Are you sure you want to delete {node_id} and all its children ?"
            )
            msg.setStandardButtons(qt.QMessageBox.Yes | qt.QMessageBox.No)
            if msg.exec_() != qt.QMessageBox.Yes:
                return

        for child in children:
            self.on_delete_item(child, showPopupForNonLeaf=False)

        branch_id = self.names.index(node_id)
        self.delete_node(self.edges[branch_id][1])

        self.names.pop(branch_id)
        self.branch_list.pop(branch_id)

        slicer.mrmlScene.RemoveNode(self.centerline_markups.pop(branch_id))
        slicer.mrmlScene.RemoveNode(self.contour_points_markups.pop(branch_id))

        self.edges.pop(branch_id)
        parent_id = self.tree_widget.getParentNodeId(node_id)
        self.tree_widget.removeNode(node_id)

        if self.current_tree_item == treeItem:
            self.current_tree_item = None

        if showPopupForNonLeaf:
            self.on_merge_only_child(parent_id)

    def on_merge_only_child(self, branch_id: str):
        """
        Merge branch with its child if it contains a single child.

        Parameters
        ----------

        branch_id: id of the branch which should be checked for merge.
        """
        if branch_id is None:
            return
        child_list = self.tree_widget.getChildrenNodeId(branch_id)
        if len(child_list) != 1:
            return

        parent_idx = self.names.index(branch_id)
        child_idx = self.names.index(child_list[0])

        # Modify parent branch to add child branch cylinders (we discard the first points because it is a common point)
        self.branch_list[parent_idx] += self.branch_list[child_idx][1:]
        self.branch_list.pop(child_idx)

        self.update_markup(parent_idx)
        slicer.mrmlScene.RemoveNode(self.centerline_markups.pop(child_idx))
        slicer.mrmlScene.RemoveNode(self.contour_points_markups.pop(child_idx))

        # Delete old child
        self.delete_node(self.edges[child_idx][0])
        self.edges[parent_idx] = self.edges[parent_idx][0], self.edges[child_idx][1]
        self.edges.pop(child_idx)
        self.names.pop(child_idx)

        self.tree_widget.removeNode(child_list[0])

    def extend_root_from_begin(
        self,
        branch: list[Cylinder],
        root_idx: int,
    ):
        """
        Extends the root from the beginning of it.


        Parameters
        ----------
        branch: the branch that will extend the root.
        root_idx: index of the root branch.
        """

        branch = branch[::-1]

        # Update the position of the new beginning of the root node
        begin_node_idx = self.edges[root_idx][0]
        self.nodes[begin_node_idx] = branch[0].center

        # Remove the last point since they have it in common
        branch = branch[:-1]

        # Concatenate
        self.branch_list[root_idx] = branch + self.branch_list[root_idx]

        # Update markups
        self.update_markup(root_idx)
