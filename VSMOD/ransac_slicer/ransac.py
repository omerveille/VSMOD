#!/usr/bin/env python-real
from .volume import Volume
from .popup_utils import CustomProgressBar, CustomStatusDialog
from .cylinder_ransac import (
    sample_around_cylinder,
    track_branch,
    Config,
)

from .cylinder import Cylinder, closest_branch
import numpy as np
from ransac_slicer.graph_branches import GraphBranches
import qt


def interpolate_point(
    cyl_0: Cylinder, cyl_1: Cylinder, vol: Volume, cfg: Config, distance: float
) -> list[Cylinder]:
    """
    Interpolate points between two cylinders.
    Points must be sparsed from at least a certain distance.

    Parameters
    ----------
    cyl_0: first cylinder to interpolate from (excluded).
    cyl_1: second cylinder to interpolate to (excluded).
    vol: the volume from which points are sampled.
    cfg: configuration regarding the RANSAC algorithm.
    distance: minimum distance between interpolated points allowed.

    Returns
    ----------

    list[Cylinder]
        a list of interpolated cylinders inbetween cyl_0 and cyl_1, both excluded.
    """
    direction = cyl_1.center - cyl_0.center
    radius_diff = cyl_1.radius - cyl_0.radius
    nb_points = int(np.linalg.norm(direction) // distance)

    cylinders = []

    centers_radius = zip(
        [
            cyl_0.center + (idx / (nb_points + 1)) * direction
            for idx in range(1, nb_points + 1)
        ],
        [
            cyl_0.radius + (idx / (nb_points + 1)) * radius_diff
            for idx in range(1, nb_points + 1)
        ],
    )
    for center, radius in centers_radius:
        cyl = Cylinder(center=center, radius=radius, direction=direction)
        cyl = sample_around_cylinder(vol, cyl, cfg)

        if cyl is not None:
            cylinders.append(cyl)

    return cylinders


def interpolate_centerline(
    cylinders: list[Cylinder],
    vol: Volume,
    cfg: Config,
    distance: float,
) -> list[Cylinder]:
    """
    Refine the points of a centerline according to a certain minimum distance between points.

    Parameters
    ----------
    cylinders: cylinders fitted through RANSAC algorithm.
    vol: the volume from which points are sampled.
    cfg: configuration regarding the RANSAC algorithm.
    distance: minimum distance between interpolated points allowed.

    Returns
    ----------

    list[Cylinder]:
        cylinders fitted through RANSAC algorithm with possibly interpolated cylinder inbetween
    """
    new_cylinders = [cylinders[0]]

    for idx in CustomProgressBar(
        iterable=range(len(cylinders) - 1),
        quantity_to_measure="segments to interpolate",
        windowTitle="Interpolating centerline points...",
        width=300,
    ):
        inbetween_cylinders = interpolate_point(
            cylinders[idx],
            cylinders[idx + 1],
            vol,
            cfg,
            distance,
        )

        new_cylinders += inbetween_cylinders
        new_cylinders.append(cylinders[idx + 1])

    return new_cylinders


def run_ransac(
    vol: Volume,
    starting_point: np.ndarray,
    direction_point: np.ndarray,
    starting_radius: float,
    percent_inlier_points: float,
    inlier_threshold: float,
    centerline_resolution: float,
    maximum_turn_angle: float,
    min_number_of_attempts: int,
    max_number_of_attempts: int,
    max_number_of_cylinders: int,
    smart_diameter_selection: bool,
    graph_branches: GraphBranches,
    progress_dialog: CustomStatusDialog,
) -> float:
    """
    Run the RANSAC algorithm to fit a cylinder according to the parameters indicated by the user.

    Apply a post-processing by refining the centerline points up to a certain resolution and updates the
    graph branch.

    Parameters
    ----------
    vol: the Volume from which points are sampled.
    starting_point: starting point of the first cylinder.
    direction_point: point indicating the direction of the first cylinder.
    starting_radius: radius in mm of the first cylinder to fit.
    percent_inlier_points: percent of inlier required to be considered a correct model.
    inlier_threshold: threshold percentage of the previous cylinder from which points are considered inlier.
    centerline_resolution: minimum distance between centerline points.
    maximum_turn_angle: the maximum turn angle possible for a vessel.
    min_number_of_attempts: the minimum number of attempts done to find a fitting cylinder.
    max_number_of_attempts: the maximum number of attempts to find a fitting cylinder.
    max_number_of_cylinders: the maximum number of cylinder tracked in one tracking.
    smart_diameter_selection: flag to indicate whether we override the radius value entered with the radius
        of the closest cylinder of the input cylinder.
    graph_branches: the graph branch object.
    progress_dialog: UI window to inform the user on the state of the branch tracking.

    Returns
    ----------

    float
        the diameter used for the first cylinder
    """
    creating_root = len(graph_branches.branch_list) == 0
    if not creating_root:
        _, _, idx_cb, idx_cyl = closest_branch(
            starting_point, graph_branches.branch_list
        )
        # We do not allow branches made out of 1 point
        if idx_cyl == len(graph_branches.branch_list[idx_cb]) - 2:
            idx_cyl = len(graph_branches.branch_list[idx_cb]) - 1

        parent_node = graph_branches.names[idx_cb]
        # Case when the closest node is the last point of a branch, we concatenate the two branches
        if idx_cyl == len(graph_branches.branch_list[idx_cb]) - 1:
            end_centerline = graph_branches.branch_list[idx_cb][idx_cyl]
        # Case when the closest node is the first point of a branch, we had the branch to the parent of the closest branch, thus the branch way have more than 2 childs
        elif idx_cyl == 0:
            parent_node = graph_branches.tree_widget.getParentNodeId(parent_node)
            end_centerline = graph_branches.branch_list[idx_cb][0]
        # Case when the closest node is in the middle of a branch, we split the branch at the intersection point
        else:
            end_centerline = graph_branches.split_branch(idx_cb, idx_cyl, parent_node)
    else:
        parent_node = None
        end_centerline = None

    direction_point = direction_point - starting_point
    # Tracking configuration
    pct_inl = percent_inlier_points / 100.0
    err = inlier_threshold / 100.0
    cfg = Config(
        percent_inliers=pct_inl,
        threshold=err,
        angle_max=maximum_turn_angle,
        nb_test_min=min_number_of_attempts,
        nb_test_max=max_number_of_attempts,
        nb_iter=max_number_of_cylinders,
    )

    # Initialize first cylinder
    if not creating_root and smart_diameter_selection:
        starting_radius = end_centerline.radius
    cyl = Cylinder(starting_point, starting_radius, direction_point, height=0)

    # Perform tracking
    cylinders = track_branch(
        vol,
        cyl,
        cfg,
        [elt for branch in graph_branches.branch_list for elt in branch],
        progress_dialog,
    )

    # Check for tracking failure
    if len(cylinders) <= 1:
        msg = qt.QMessageBox()
        msg.setIcon(qt.QMessageBox.Critical)
        msg.setWindowTitle("Error")
        msg.setText("Could not find any branch")
        msg.exec_()
        graph_branches.on_merge_only_child(parent_node)
        return starting_radius * 2

    # Add the first node if we just created the root
    if creating_root:
        graph_branches.nodes.append(cylinders[0].center)
    else:
        # We add the closest point to the cylinders list so the space inbetween can be interpolated aswell
        cylinders.insert(0, end_centerline)

    # Interpolate points
    cylinders = interpolate_centerline(
        cylinders,
        vol,
        cfg,
        distance=centerline_resolution,
    )

    # Case of a split branch / new root / concatenation of branches
    if creating_root or parent_node is not None:
        graph_branches.nodes.append(cylinders[-1].center)
        edge_begin = (
            graph_branches.edges[graph_branches.names.index(parent_node)][1]
            if not creating_root
            else len(graph_branches.nodes) - 2
        )
        graph_branches.create_new_branch(
            (edge_begin, len(graph_branches.nodes) - 1),
            cylinders,
            parent_node,
        )
    # Case when we are creating a new branch extending the root before the initial point (the only case of backward extension)
    else:
        graph_branches.extend_root_from_begin(cylinders, idx_cb)
    return starting_radius * 2
