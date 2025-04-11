from typing import Union
from .popup_utils import CustomProgressBar, CustomStatusDialog
from .cylinder import Cylinder
import slicer
import numpy as np
import vtk
from skimage.morphology import binary_dilation, ball
from .color_palettes import vessel_colors, contour_color
import time
import networkx as nx


def _compute_draw_order(
    graph_branches_nodes: list[np.ndarray], graph_branches_edges: list[tuple[int, int]]
) -> list[int]:
    """
    Define the order in which the branches will be drawn.
    We draw in reverse bfs order, so that parent branches are always drawn
    on top of childs.

    Parameters
    ----------

    graph_branches_nodes: list of points which are the birfucation, root or leaves
    graph_branches_edges: list of all the edges

    Returns
    ----------

    list[int]:
        indexes of the branch to draw. Drawn from first to last listed.
    """

    G = nx.DiGraph()
    for i, node in enumerate(graph_branches_nodes):
        G.add_node(i, pos=node)

    for i, edge in enumerate(graph_branches_edges):
        G.add_edge(
            edge[0],
            edge[1],
            edge_idx=i,
        )
    return [G[a][b]["edge_idx"] for a, b in nx.bfs_edges(G, source=0)][::-1]


def _compute_bbox(
    centerline_points: list[np.ndarray],
    radius: list[np.ndarray],
    lower_bound: np.ndarray,
    dimensions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the bounding box of a zone on which to evaluate a sphere equation.

    The bounding box cannot be located outside of the volume bounds.

    Parameters
    ----------

    centerline_points: list of sphere centers.
    radius: list of the sphere radius.
    lower_bound: lower coordinate limit on which point can be located.
    dimensions: volume dimensions.

    Returns
    ----------

    min_point, max_point the corners of the bounding box.
    """
    centerline_points = np.array(centerline_points)
    radius = np.array(radius)

    min_point = np.maximum(
        np.floor(np.min(centerline_points - radius, axis=0)).astype(int), lower_bound
    )
    max_point = np.minimum(
        np.ceil(np.max(centerline_points + radius, axis=0)).astype(int), dimensions
    )

    return min_point, max_point


def _adapt_radius(
    radius: float, reduction_threshold: float, reduction_factor: float
) -> float:
    """
    Reduce the growth of big vessels radius to limit the leaking when creating
    segmentation zones.

    Parameters
    ----------

    radius: a sphere radius.
    reduction_threshold: threshold from which a reduction is applied to the radius.
    reduction_factor: amount of the reduction.

    Returns
    ----------

    Newly adapted radius.
    """

    return (
        radius
        if radius <= reduction_threshold
        else (radius - reduction_threshold) * reduction_factor + reduction_threshold
    )


def _split_list(lst, n):
    """
    Split list in n parts.
    """
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def update_segment(
    segment_ids: Union[str, list],
    labelmap_node: slicer.vtkMRMLSegmentationNode,
    data: np.ndarray,
    segmentation_node: slicer.vtkMRMLLabelMapVolumeNode,
):
    """
    Update a segment labelmap.

    Parameters
    ----------

    segment_ids: id or list of ids of the segment(s) to update.
    labelmap_node: slicer labelmap to update.
    data: numpy array that delimit the segment zone.
    segmentation_node: slicer segmentation on which the segment are updated.
    """
    vtk_segment_id = vtk.vtkStringArray()
    if isinstance(segment_ids, list):
        for ids in segment_ids:
            vtk_segment_id.InsertNextValue(ids)
    else:
        vtk_segment_id.InsertNextValue(segment_ids)

    slicer.util.updateVolumeFromArray(labelmap_node, data)
    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
        labelmap_node, segmentation_node, vtk_segment_id
    )


def paint_segments(
    volume_node: slicer.vtkMRMLScalarVolumeNode,
    branches: list[list[Cylinder]],
    centerline_names: list[str],
    nodes: list[np.ndarray],
    edges: list[tuple[int, int]],
    segmentation_node: slicer.vtkMRMLSegmentationNode,
    reduction_factor: float,
    radius_reduction_threshold: float,
    contour_distance: int,
    merge_all_vessels: bool,
) -> np.ndarray:
    """
    Paint the segmentations segments according to the centerlines and their associated radius.

    Parameters
    ----------

    volume_node: input volume.
    branches: list of all the branches segmented.
    centerline_names: list of the names of the centerlines.
    nodes: lists of the nodes of the branches graph.
    edges: lists of the edges of the branches graph.
    segmentation_node: segmentation_node on which segment will be added and updated.
    reduction_factor: amount of the reduction.
    radius_reduction_threshold: threshold from which a reduction is applied to the radius.
    contour_distance: distance in voxel between the vessel and the contour.
    merge_all_vessels: if True, this flag will put every vessels in the same segment, instead of separeted ones.

    Returns
    ----------

    The segmentation mapping of the volume.
    """

    # Extract the centerline information from the branches
    centerlines = []
    radius = []
    for branch in CustomProgressBar(
        iterable=branches,
        quantity_to_measure="branches processed",
        windowTitle="Applying preprocessing to branches ...",
        width=300,
    ):
        centerline = []
        centerline_radius = []
        for cyl in branch:
            centerline.append(cyl.center)
            # We underestimate the radius of the cylinder in order to not make the seed leak out of the vessels
            underestimated_radius = np.linalg.norm(
                np.array(cyl._contour_points) - cyl.center,
                axis=1,
            ).min()
            centerline_radius.append(underestimated_radius)
        centerlines.append(np.array(centerline, dtype=np.float64))
        radius.append(centerline_radius)

    # Compute the draw order
    branch_draw_order = _compute_draw_order(nodes, edges)

    # Important variables
    voxel_spacing = np.array(volume_node.GetSpacing()[::-1])
    volume_dimensions = np.array(volume_node.GetImageData().GetDimensions()[::-1])
    segmentation = segmentation_node.GetSegmentation()

    # Get the ras to ijk matrix as numpy array
    ras_to_ijk = vtk.vtkMatrix4x4()
    volume_node.GetRASToIJKMatrix(ras_to_ijk)
    np_ras_to_ijk = np.zeros(shape=(4, 4))
    ras_to_ijk.DeepCopy(np_ras_to_ijk.ravel(), ras_to_ijk)

    # Clear all segments of segmentation
    progress_dialog = CustomStatusDialog(
        windowTitle="Clearing all segments ...",
        text="Please wait",
        width=300,
        height=50,
    )
    # Timer added to let the interface load
    time.sleep(0.1)

    progress_dialog.set_text("Clearing all segments ...")
    segmentation.RemoveAllSegments()
    progress_dialog.close()

    # Ensure the labelmap has the same dimensions, spacing, orientation, localisation as the volume
    labelmap_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    labelmap_node.CopyOrientation(volume_node)
    labelmap_node.SetOrigin(volume_node.GetOrigin())
    labelmap_node.SetSpacing(voxel_spacing)
    ijk_to_ras = vtk.vtkMatrix4x4()
    volume_node.GetIJKToRASMatrix(ijk_to_ras)
    labelmap_node.SetIJKToRASMatrix(ijk_to_ras)

    labelmap_node.CreateDefaultDisplayNodes()
    labelmap_node.SetAndObserveImageData(vtk.vtkImageData())
    labelmap_node.GetImageData().SetDimensions(volume_dimensions)

    # Set the labelmap pixel values to uint16 with one channel (65535 max segment)
    labelmap_node.GetImageData().AllocateScalars(vtk.VTK_UNSIGNED_SHORT, 1)

    # Transform ras coordinates (real world coordinates) into ijk coordinates (voxel coordinates)
    centerlines = [
        [
            [(np_ras_to_ijk @ np.array([*point, 1]))[-2::-1] for point in list_part]
            for list_part in _split_list(centerline, 3)
        ]
        for centerline in centerlines
    ]
    radius = [list(_split_list(radius_list, 3)) for radius_list in radius]

    # Constants
    x, y, z = np.indices(volume_dimensions)
    low_bound = np.array([0, 0, 0], dtype=int)
    high_bound = volume_dimensions - 1

    # Radius is pixel coordinates
    ijk_radius_thresholded = [
        [
            [
                np.array(
                    [_adapt_radius(r, radius_reduction_threshold, reduction_factor)] * 3
                )
                / voxel_spacing
                for r in radius_part
            ]
            for radius_part in radius_per_centerline
        ]
        for radius_per_centerline in radius
    ]
    segment_map = np.zeros(volume_dimensions, dtype=np.uint16)

    """
    Paint segment for each centerline, the segment region is underestimated for large vessels (c.f _adapt_radius function) so that
    it does not leak outside the vessels
    """
    for centerline_idx in CustomProgressBar(
        iterable=branch_draw_order,
        quantity_to_measure="vessels painted",
        windowTitle="Computing segment regions...",
        width=300,
    ):
        centerline = centerlines[centerline_idx]
        radius_per_centerline = ijk_radius_thresholded[centerline_idx]
        sub_segment_map = np.zeros_like(segment_map, dtype=np.bool_)
        for points, points_radius in zip(centerline, radius_per_centerline):
            lower_edge, highter_edge = _compute_bbox(
                points, points_radius, low_bound, volume_dimensions
            )
            for point, radius_ in zip(points, points_radius):
                center_x, center_y, center_z = point
                radius_x, radius_y, radius_z = radius_

                sphere_map = (
                    (
                        (
                            x[
                                lower_edge[0] : highter_edge[0],
                                lower_edge[1] : highter_edge[1],
                                lower_edge[2] : highter_edge[2],
                            ]
                            - center_x
                        )
                        ** 2
                        / radius_x**2
                        + (
                            y[
                                lower_edge[0] : highter_edge[0],
                                lower_edge[1] : highter_edge[1],
                                lower_edge[2] : highter_edge[2],
                            ]
                            - center_y
                        )
                        ** 2
                        / radius_y**2
                        + (
                            z[
                                lower_edge[0] : highter_edge[0],
                                lower_edge[1] : highter_edge[1],
                                lower_edge[2] : highter_edge[2],
                            ]
                            - center_z
                        )
                        ** 2
                        / radius_z**2
                    )
                    <= 1
                ).astype(np.bool_)
                closest_pixel_to_paint = np.maximum(
                    np.minimum(
                        [round(coord) for coord in point], high_bound, dtype=int
                    ),
                    low_bound,
                    dtype=int,
                )

                sub_segment_map[
                    closest_pixel_to_paint[0],
                    closest_pixel_to_paint[1],
                    closest_pixel_to_paint[2],
                ] = True
                sub_segment_map[
                    lower_edge[0] : highter_edge[0],
                    lower_edge[1] : highter_edge[1],
                    lower_edge[2] : highter_edge[2],
                ] += sphere_map

        segment_map[sub_segment_map] = centerline_idx + 1

    del ijk_radius_thresholded

    # Paint contours will be a dilated version of the already computed segment map
    contours_map = (segment_map > 0).astype(np.bool_)
    # Filtering point to keep the one that have been shrunk, we will compute the inside region but not underestimated this time
    centerline_and_radius = [
        [centerline_part, [np.array([r] * 3) / voxel_spacing for r in radius_part]]
        for (centerline, radius_per_centerline) in zip(centerlines, radius)
        for (centerline_part, radius_part) in zip(centerline, radius_per_centerline)
        if any(map(lambda x: x > radius_reduction_threshold, radius_part))
    ]

    for points, points_radius in centerline_and_radius:
        lower_edge, highter_edge = _compute_bbox(
            points, points_radius, low_bound, volume_dimensions
        )
        for point, radius_ in zip(points, points_radius):
            center_x, center_y, center_z = point
            radius_x, radius_y, radius_z = radius_

            sphere_map = (
                (
                    (
                        x[
                            lower_edge[0] : highter_edge[0],
                            lower_edge[1] : highter_edge[1],
                            lower_edge[2] : highter_edge[2],
                        ]
                        - center_x
                    )
                    ** 2
                    / radius_x**2
                    + (
                        y[
                            lower_edge[0] : highter_edge[0],
                            lower_edge[1] : highter_edge[1],
                            lower_edge[2] : highter_edge[2],
                        ]
                        - center_y
                    )
                    ** 2
                    / radius_y**2
                    + (
                        z[
                            lower_edge[0] : highter_edge[0],
                            lower_edge[1] : highter_edge[1],
                            lower_edge[2] : highter_edge[2],
                        ]
                        - center_z
                    )
                    ** 2
                    / radius_z**2
                )
                <= 1
            ).astype(np.bool_)
            contours_map[
                lower_edge[0] : highter_edge[0],
                lower_edge[1] : highter_edge[1],
                lower_edge[2] : highter_edge[2],
            ] += sphere_map

    progress_dialog = CustomStatusDialog(
        windowTitle="Computing contours ...", text="Please wait", width=300, height=50
    )
    # Timer added to let the interface load
    time.sleep(0.1)

    # Outter edge
    progress_dialog.set_text("Computing the outter edge ...")
    # It's technicly incorrect to use a ball, as voxels are not perfect cubes, but well if it works fine ...
    contours_dilated = binary_dilation(contours_map, ball(radius=contour_distance + 2))

    # Inner edge
    progress_dialog.set_text("Computing the inner edge ...")
    contours_dilated[
        binary_dilation(contours_map, ball(radius=contour_distance))
    ] = False
    del contours_map

    # Add the segment to the segmentation, also merge the segment and contour map
    ids = []
    if merge_all_vessels:
        segment_map = (segment_map > 0).astype(np.uint16) + contours_dilated.astype(
            np.uint16
        ) * 2
        ids.append(segmentation.AddEmptySegment("", "Vessels", vessel_colors[0]))
    else:
        segment_map += contours_dilated.astype(np.uint16) * (len(centerlines) + 1)
        for idx, segment_name in enumerate(centerline_names):
            ids.append(
                segmentation.AddEmptySegment(
                    "", segment_name, vessel_colors[idx % len(vessel_colors)]
                )
            )
    del contours_dilated

    # Add contours to segment id list, also makes it transparent
    ids.append(segmentation.AddEmptySegment("", "Contours", contour_color))
    segmentation_node.GetDisplayNode().SetSegmentOpacity3D(ids[-1], 0.1)

    # Write the labelmap inside the segmentation
    update_segment(
        ids,
        labelmap_node,
        segment_map,
        segmentation_node,
    )

    slicer.mrmlScene.RemoveNode(labelmap_node)
    progress_dialog.close()

    return segment_map
