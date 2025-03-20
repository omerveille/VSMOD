import importlib
import sys
from typing import Annotated, Optional

import numpy as np
import qt
import slicer
import slicer.util
import vtk
import json
from math import radians

from slicer import (
    vtkMRMLScalarVolumeNode,
    vtkMRMLMarkupsFiducialNode,
    vtkMRMLMarkupsLineNode,
)
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModule,
    ScriptedLoadableModuleTest,
)
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)
from slicer.util import VTKObservationMixin

try:
    # Import path working for Slicer 5.6.1
    from vtkSlicerMarkupsModuleMRMLPython import vtkMRMLMarkupsNode
except ImportError:
    # Import path working for Slicer 5.9.0
    from slicer import vtkMRMLMarkupsNode

# Recursive reload, when you hit the "reload" button in 3D slicer, force all submodules to be reloaded (which is not the case by default).
try:
    to_reload = [key for key in sys.modules.keys() if key.startswith("ransac_slicer")]
    to_reload.sort(key=len, reverse=True)
    for file_to_reload in to_reload:
        sys.modules[file_to_reload] = importlib.reload(sys.modules[file_to_reload])
except Exception as e:
    print(f"Exception occurred while reloading\n{e}")

from ransac_slicer.dependencies_checker import check_and_install_missing_dependencies
from ransac_slicer.ransac import run_ransac
from ransac_slicer.graph_branches import GraphBranches
from ransac_slicer.branch_tree import BranchTree, TreeColumnRole, Icons
from ransac_slicer.color_palettes import direction_points_color
from ransac_slicer.popup_utils import (
    CustomStatusDialog,
)
from ransac_slicer.volume import Volume
from ransac_slicer.region_growing_seeds import paint_segments, _compute_draw_order
from ransac_slicer.jit_compiled_functions import numba_close

from networkx.readwrite import json_graph
import networkx as nx

#
# pulmonary_arteries_segmentor_module
#


class pulmonary_arteries_segmentor_module(ScriptedLoadableModule):
    """
    Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Pulmonary Arteries Segmentor"
        self.parent.categories = ["Segmentation"]
        self.parent.dependencies = []
        self.parent.contributors = ["Azéline Aillet", "Gabriel Husak"]
        self.parent.helpText = """
A 3D Slicer plugin for pulmonary artery extraction from angiography images.
"""
        self.parent.acknowledgementText = """
This plugin is an end-of-study project, made by Azéline Aillet (Student at EPITA) and Gabriel Husak (Student at EPITA), under the direction of Odyssée Merveille (CREATIS) and Morgane Des Ligneris (CREATIS).
The RANSAC code is based on the previous work of Jack CARBONERO (CReSTIC), Guillaume DOLLE (LMR) and Nicolas PASSAT (CReSTIC) on the plugin vestract.
The hierarchy code is based on the work of Lucie Macron (Kitware SAS), Thibault Pelletier (Kitware SAS), Camille Huet (Kitware SAS), Leo Sanchez (Kitware SAS) from the RVesselX plugin.
"""


#
# pulmonary_arteries_segmentor_moduleParameterNode
#


@parameterNodeWrapper
class pulmonary_arteries_segmentor_moduleParameterNode:
    """
    Class to wrap the inputs of the plugin's interface, the fields are automaticaly updated.

    Fields
    ----------

    inputVolume: input volume to extract the arteries from.
    startingPoint: starting point list for RANSAC cylinders.
    directionPoint: direction point list for RANSAC cylinders.

    startingRadius: initiale radius of the first cylinder to fit.
    centerlineResolution: maximum allowed distance between to point on the tracked centerline.

    percentInlierPoints: percentage of inlier points to validate a cylinder.
    percentThreshold: percentage of last cylinders radius to make a point inlier of a cylinder.
    maximumTurnAngle: the maximum possible angle between two consecutive cylinder
    maxNumberOfAttempts: the maximum amount of candidate cylinder to test
    maxNumberOfCylinders: the maximum amount of cylinder tracked in one branch
    """

    # Input tab
    inputVolume: vtkMRMLScalarVolumeNode
    startingPoint: vtkMRMLMarkupsFiducialNode
    directionPoint: vtkMRMLMarkupsFiducialNode
    measureDistance: vtkMRMLMarkupsLineNode

    # Simple Ransac paramaters
    startingRadius: Annotated[float, WithinRange(0.1, 1000.0)] = 10.0
    centerlineResolution: Annotated[float, WithinRange(0.1, 1000.0)] = 5.0

    # Advanced Ransac paramaters
    percentInlierPoints: Annotated[float, WithinRange(0.0, 100.0)] = 60.0
    percentThreshold: Annotated[float, WithinRange(0.0, 100.0)] = 30.0
    maximumTurnAngle: Annotated[float, WithinRange(0.0, 90.0)] = 60.0
    minNumberOfAttempts: Annotated[int, WithinRange(0, 99999999)] = 0
    maxNumberOfAttempts: Annotated[int, WithinRange(0, 99999999)] = 1000
    maxNumberOfCylinders: Annotated[int, WithinRange(1, 99999999)] = 1000

    # Segmentation parameters
    reductionFactor: Annotated[float, WithinRange(0.0, 1.0)] = 0.75
    reductionThreshold: Annotated[float, WithinRange(0.0, 1000.0)] = 5.0
    contourDistance: Annotated[int, WithinRange(1, 1000)] = 4


#
# pulmonary_arteries_segmentor_moduleWidget
#


class pulmonary_arteries_segmentor_moduleWidget(
    ScriptedLoadableModuleWidget, VTKObservationMixin
):
    """
    Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """
        Called when the user opens the module the first time and the widget is initialized.
        """

        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self.parameterNode = None
        self.parameterNodeGuiTag = None
        self.graph_branches = None
        self.segmentationNode = None
        self.nodeDeletionObserverTag = None
        self.isPlacingPoints = False

    def setup(self) -> None:
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Check slicer version
        slicer_version = (
            slicer.app.majorVersion,
            slicer.app.minorVersion,
            float(slicer.app.revision),
        )
        if slicer_version < (5, 6, 1):
            error_msg = (
                "This plugin is only compatible for slicer version superior to 5.6.1.\n"
                "Please download the latest Slicer version to use this plugin."
            )
            self.layout.addWidget(qt.QLabel(error_msg))
            self.layout.addStretch()
            slicer.util.errorDisplay(error_msg)
            return

        check_and_install_missing_dependencies()

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(
            self.resourcePath("UI/pulmonary_arteries_segmentor_module.ui")
        )
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        segmentEditorSingletonTag = "SegmentEditor"
        segmentEditorNode = slicer.mrmlScene.GetSingletonNode(
            segmentEditorSingletonTag, "vtkMRMLSegmentEditorNode"
        )
        if segmentEditorNode is None:
            segmentEditorNode = slicer.mrmlScene.CreateNodeByClass(
                "vtkMRMLSegmentEditorNode"
            )
            segmentEditorNode.UnRegister(None)
            segmentEditorNode.SetSingletonTag(segmentEditorSingletonTag)
            segmentEditorNode = slicer.mrmlScene.AddNode(segmentEditorNode)
        if (
            not hasattr(self, "segmentEditorNode")
            or self.segmentEditorNode != segmentEditorNode
        ):
            self.segmentEditorNode = segmentEditorNode
            self.ui.SegmentEditorWidget.setMRMLSegmentEditorNode(self.segmentEditorNode)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = pulmonary_arteries_segmentor_moduleLogic()

        self.branch_tree = BranchTree()
        begin_tab = self.ui.tabWidget.widget(0)

        # Insert the branch tree widget defined in code
        begin_tab.layout().insertWidget(6, self.branch_tree)

        self.graph_branches = GraphBranches(
            self.branch_tree,
            self.ui.showCenterlineButton,
            self.ui.showContourPointsButton,
            self.ui.lockButton,
        )
        # Connections / Callbacks

        # These connections ensure that we update parameter node when scene is closed
        self._addObserver(
            slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose
        )
        self._addObserver(
            slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose
        )

        # Sliders callbacks
        self.ui.centerlineTextSize.connect("valueChanged(double)", self.changeTextSize)

        # Buttons callbacks
        self.ui.placePointButton.connect("clicked(bool)", self.startPlacePointProcedure)
        self.ui.measureRadiusButton.connect("clicked(bool)", self.measure)

        self.ui.createBranch.connect("clicked(bool)", self.create_branch)
        self.ui.clearTree.connect(
            "clicked(bool)",
            lambda: (
                self.graph_branches.clear_all(),
                self.checkCanStartSegmentation(),
                self.checkCanStartRansac(),
            ),
        )

        self.ui.exportTreeButton.connect(
            "clicked(bool)", self.graph_branches.save_networkX
        )
        self.ui.loadTreeArchitectureButton.connect(
            "clicked(bool)", self.onLoadTreeArchitecture
        )

        self.ui.paintButton.connect("clicked(bool)", self.onStartSegmentationButton)

        self.ui.lockButton.connect("clicked(bool)", self.onLockButton)
        self.ui.showCenterlineButton.connect(
            "clicked(bool)",
            lambda: self.graph_branches.on_header_clicked(
                TreeColumnRole.VISIBILITY_CENTER
            ),
        )
        self.ui.showContourPointsButton.connect(
            "clicked(bool)",
            lambda: self.graph_branches.on_header_clicked(
                TreeColumnRole.VISIBILITY_CONTOUR
            ),
        )

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self) -> None:
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def enter(self) -> None:
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        if hasattr(self, "logic") and self.logic is not None:
            self.initializeParameterNode()

    def exit(self) -> None:
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self.parameterNode:
            self.parameterNode.disconnectGui(self.parameterNodeGuiTag)
            self.parameterNodeGuiTag = None
            self._removeObserver(
                self.parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self.checkCanStartRansac,
            )

    def _addObserver(self, obj, event, fct):
        """
        Wrapper of addObserver function, does nothing if the observer already exists.
        """
        if not self.hasObserver(obj, event, fct):
            self.addObserver(obj, event, fct)

    def _removeObserver(self, obj, event, fct):
        """
        Wrapper of removeObserver function, does nothing if the observer does not exists.
        """
        if self.hasObserver(obj, event, fct):
            self.removeObserver(obj, event, fct)

    def onSceneStartClose(self, caller, event) -> None:
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """
        Ensure parameter node exists and are observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self.parameterNode.inputVolume:
            node = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if node:
                self.parameterNode.inputVolume = node

        # Create starting and direction point parameters if they do not already exist and select them
        if not self.parameterNode.startingPoint:
            node = slicer.util.getFirstNodeByClassByName(
                "vtkMRMLMarkupsFiducialNode", "s"
            )
            if node is None or not isinstance(node, slicer.vtkMRMLMarkupsFiducialNode):
                node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
                node.SetName("s")
            node.GetDisplayNode().SetSelectedColor(*direction_points_color)
            self.parameterNode.startingPoint = node

        if not self.parameterNode.directionPoint:
            node = slicer.util.getFirstNodeByClassByName(
                "vtkMRMLMarkupsFiducialNode", "d"
            )
            if node is None or not isinstance(node, slicer.vtkMRMLMarkupsFiducialNode):
                node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
                node.SetName("d")
            node.GetDisplayNode().SetSelectedColor(*direction_points_color)
            self.parameterNode.directionPoint = node

        # Create a line to measure diameters if it does not already exist
        if not self.parameterNode.measureDistance:
            node = slicer.util.getFirstNodeByClassByName(
                "vtkMRMLMarkupsLineNode", "distance"
            )
            if node is None or not isinstance(node, slicer.vtkMRMLMarkupsLineNode):
                node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode")
                node.SetName("distance")
            self.parameterNode.measureDistance = node

    def setParameterNode(
        self,
        inputParameterNode: Optional[pulmonary_arteries_segmentor_moduleParameterNode],
    ) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self.parameterNode:
            self.parameterNode.disconnectGui(self.parameterNodeGuiTag)
            self._removeObserver(
                self.parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self.checkCanStartRansac,
            )
            self._removeObserver(
                self.parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self.checkCanStartSegmentation,
            )
            self._removeObserver(
                self.parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self.checkCanPlacePoint,
            )
            self._removeObserver(
                self.parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self.checkCanMeasure,
            )
        self.parameterNode = inputParameterNode
        if self.parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self.parameterNodeGuiTag = self.parameterNode.connectGui(self.ui)
            self._addObserver(
                self.parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self.checkCanStartRansac,
            )
            self._addObserver(
                self.parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self.checkCanStartSegmentation,
            )
            self._addObserver(
                self.parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self.checkCanPlacePoint,
            )
            self._addObserver(
                self.parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self.checkCanMeasure,
            )
            self.checkCanStartRansac()
            self.checkCanPlacePoint()
            self.checkCanMeasure()

    def checkCanStartRansac(self, caller=None, event=None) -> None:
        """
        Update the create and delete button state depending on the parameters state.
        """
        starting_point = self.parameterNode.startingPoint
        direction_point = self.parameterNode.directionPoint

        if starting_point:
            self._addObserver(
                starting_point,
                vtkMRMLMarkupsNode.PointAddedEvent,
                self.checkCanStartRansac,
            )
            self._addObserver(
                starting_point,
                vtkMRMLMarkupsNode.PointRemovedEvent,
                self.checkCanStartRansac,
            )

        if direction_point:
            self._addObserver(
                direction_point,
                vtkMRMLMarkupsNode.PointAddedEvent,
                self.checkCanStartRansac,
            )
            self._addObserver(
                direction_point,
                vtkMRMLMarkupsNode.PointRemovedEvent,
                self.checkCanStartRansac,
            )

        if (
            self.parameterNode
            and not self.isPlacingPoints
            and all(
                [
                    self.parameterNode.inputVolume,
                    self.parameterNode.startingPoint,
                    self.parameterNode.directionPoint,
                ]
            )
            and starting_point.GetNumberOfControlPoints()
            and direction_point.GetNumberOfControlPoints()
        ):
            self.ui.createBranch.enabled = True
            if len(self.graph_branches.names) == 0:
                self.ui.createBranch.text = "Create Root"
                self.ui.createBranch.toolTip = "Create Root."
                self.ui.useLastTrackedRadius.enabled = False
                self.ui.useLastTrackedRadius.checked = False
            else:
                self.ui.createBranch.text = "Create New Branch"
                self.ui.createBranch.toolTip = "Create New Branch."
                self.ui.useLastTrackedRadius.enabled = True
        else:
            self.ui.createBranch.toolTip = "Select all input before creating branch."
            self.ui.createBranch.enabled = False

        if len(self.graph_branches.names) != 0:
            self.ui.clearTree.toolTip = "Clear all tree."
            self.ui.clearTree.enabled = True
            self.ui.exportTreeButton.toolTip = "Export the network X graph of the centerlines and contour points as JSON and pickle."
            self.ui.exportTreeButton.enabled = True
        else:
            self.ui.clearTree.toolTip = "Tree is already empty."
            self.ui.clearTree.enabled = False
            self.ui.exportTreeButton.toolTip = "There is nothing to save."
            self.ui.exportTreeButton.enabled = False

    def checkCanPlacePoint(self, *args):
        """
        Enables the place button if the starting and direction point parameters exist.
        """
        id_starting = (
            self.parameterNode.startingPoint.GetID()
            if self.parameterNode.startingPoint
            else None
        )
        id_direction = (
            self.parameterNode.directionPoint.GetID()
            if self.parameterNode.directionPoint
            else None
        )

        self.ui.placePointButton.enabled = (
            id_starting and id_direction and (id_starting != id_direction)
        )

    def startPlacePointProcedure(self, *args):
        """
        Start a point placement procedure.

        Each function of this procedure are called through observers callbacks.
        """
        self.startingPointPlaced = False
        self.directionPointPlaced = False
        self.isPlacingPoints = True
        self.ui.createBranch.enabled = False

        starting_point = self.parameterNode.startingPoint
        starting_point.GetDisplayNode().SetSelectedColor(*direction_points_color)

        # Prepare the case where the user place the first point
        self._addObserver(
            starting_point,
            vtkMRMLMarkupsNode.PointPositionDefinedEvent,
            self.directionPointPlacement,
        )

        # Prepare the case where the user cancel the point placement
        self._addObserver(
            starting_point,
            vtkMRMLMarkupsNode.PointRemovedEvent,
            self.resetPlacementState,
        )

        # Start placing procedure
        slicer.app.applicationLogic().GetSelectionNode().SetActivePlaceNodeID(
            starting_point.GetID()
        )
        slicer.modules.markups.logic().StartPlaceMode(1)

    def directionPointPlacement(self, *args):
        """
        First state of a point placement procedure.

        Each function of this procedure are called through observers callbacks.
        """
        self.startingPointPlaced = True
        starting_point = self.parameterNode.startingPoint

        if not starting_point:
            return

        self._removeObserver(
            starting_point,
            vtkMRMLMarkupsNode.PointPositionDefinedEvent,
            self.directionPointPlacement,
        )
        self._removeObserver(
            starting_point,
            vtkMRMLMarkupsNode.PointRemovedEvent,
            self.resetPlacementState,
        )

        direction_point = self.parameterNode.directionPoint
        direction_point.GetDisplayNode().SetSelectedColor(*direction_points_color)

        # Prepare the case where the user placed the last point
        self._addObserver(
            direction_point,
            vtkMRMLMarkupsNode.PointPositionDefinedEvent,
            self.validate_last_point,
        )

        # Prepare the case where the user cancel the point placement
        self._addObserver(
            direction_point,
            vtkMRMLMarkupsNode.PointRemovedEvent,
            self.resetPlacementState,
        )

        # Place direction point
        slicer.app.applicationLogic().GetSelectionNode().SetActivePlaceNodeID(
            direction_point.GetID()
        )
        slicer.modules.markups.logic().StartPlaceMode(1)

    def validate_last_point(self, *args):
        """
        Finish a point placement procedure.

        Each function of this procedure are called through observers callbacks.
        """
        self.directionPointPlaced = True
        slicer.modules.markups.logic().StartPlaceMode(0)
        self.resetPlacementState()

    def resetPlacementState(self, *args):
        """
        Reset the state of the point placement procedure.

        Meaning if the user placed only half of points or cancels point placement,
        put the system into a stable state.

        If all points have been placed, remove the extra points.

        Each function of this procedure are called through observers callbacks.
        """
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()

        if (
            interactionNode.GetPlaceModePersistence() == 1
            and interactionNode.GetCurrentInteractionMode() == 1
        ):
            # We do not go further because if those conditions are met, it means that the user moved cursor out of window
            return

        self.isPlacingPoints = False
        self.checkCanStartRansac()

        starting_point = self.parameterNode.startingPoint
        self._removeObserver(
            starting_point,
            vtkMRMLMarkupsNode.PointPositionDefinedEvent,
            self.directionPointPlacement,
        )
        self._removeObserver(
            starting_point,
            vtkMRMLMarkupsNode.PointRemovedEvent,
            self.resetPlacementState,
        )

        direction_point = self.parameterNode.directionPoint
        self._removeObserver(
            direction_point,
            vtkMRMLMarkupsNode.PointPositionDefinedEvent,
            self.validate_last_point,
        )
        self._removeObserver(
            direction_point,
            vtkMRMLMarkupsNode.PointRemovedEvent,
            self.resetPlacementState,
        )

        if self.startingPointPlaced and not self.directionPointPlaced:
            starting_point.RemoveNthControlPoint(
                starting_point.GetNumberOfControlPoints() - 1
            )

        while starting_point.GetNumberOfControlPoints() >= 2:
            starting_point.RemoveNthControlPoint(0)

        while direction_point.GetNumberOfControlPoints() >= 2:
            direction_point.RemoveNthControlPoint(0)

        self.startingPointPlaced = False
        self.directionPointPlaced = False

    def checkCanMeasure(self, *args):
        """
        Enables the measure button if the distance parameter exist.
        """
        self.ui.measureRadiusButton.enabled = (
            self.parameterNode.measureDistance is not None
        )

    def measure(self, *args):
        """
        Start a measuring procedure.

        Each function of this procedure are called through observers callbacks.
        """
        measuring_node = self.parameterNode.measureDistance

        if not measuring_node:
            return

        measuring_node.RemoveAllControlPoints()

        self._addObserver(
            measuring_node,
            vtkMRMLMarkupsNode.PointPositionDefinedEvent,
            self.measureNodePlaced,
        )
        self._addObserver(
            measuring_node,
            vtkMRMLMarkupsNode.PointRemovedEvent,
            self.measureNodeRemoved,
        )

        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        selectionNode.SetReferenceActivePlaceNodeClassName("vtkMRMLMarkupsLineNode")
        selectionNode.SetActivePlaceNodeID(measuring_node.GetID())

        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        interactionNode.SetCurrentInteractionMode(slicer.vtkMRMLInteractionNode.Place)

        slicer.app.applicationLogic().GetSelectionNode().SetActivePlaceNodeID(
            measuring_node.GetID()
        )

    def measureNodePlaced(self, *args):
        """
        Callback when a point of the measuring procedure is placed.
        If both point of the line are placed, enter the measured real world distance between points
        in the UI radius field.

        Each function of this procedure are called through observers callbacks.
        """
        measuring_node = self.parameterNode.measureDistance

        if not measuring_node:
            return

        if measuring_node.GetNumberOfControlPoints() == 2:
            distance = measuring_node.GetLineLengthWorld()
            if numba_close(distance, 0.0):
                distance = 0.2
            self.parameterNode.startingRadius = distance / 2

    def measureNodeRemoved(self, *args):
        """
        Callback when a point of the measuring procedure is removed. (eg when we cancel the point placement)
        In this case we remove all the points of the measuring line.

        Each function of this procedure are called through observers callbacks.
        """
        measuring_node = self.parameterNode.measureDistance

        if not measuring_node:
            return

        measuring_node.RemoveAllControlPoints()

    def recenter3dView(self) -> None:
        """
        Recenter the 3D slicer's 3D view on the subject.
        """
        layoutManager = slicer.app.layoutManager()
        threeDWidget = layoutManager.threeDWidget(0)
        threeDView = threeDWidget.threeDView()
        threeDView.resetFocalPoint()

    def create_branch(self) -> None:
        """
        Start the RANSAC algorithm according to user input parameters.
        """
        with slicer.util.tryWithErrorDisplay(
            "Failed to compute tracking.", waitCursor=True
        ):
            progress_dialog = CustomStatusDialog(
                windowTitle="Computing centerline...",
                text="Please wait",
                width=300,
                height=50,
            )
            self.graph_branches = self.logic.processBranch(
                raw_volume=self.parameterNode.inputVolume,
                starting_point_list=self.parameterNode.startingPoint,
                direction_point_list=self.parameterNode.directionPoint,
                percent_inlier_points=self.parameterNode.percentInlierPoints,
                inlier_threshold=self.parameterNode.percentThreshold,
                starting_radius=self.parameterNode.startingRadius,
                centerline_resolution=self.parameterNode.centerlineResolution,
                maximum_turn_angle=self.parameterNode.maximumTurnAngle,
                min_number_of_attempts=self.parameterNode.minNumberOfAttempts,
                max_number_of_attempts=self.parameterNode.maxNumberOfAttempts,
                max_number_of_cylinders=self.parameterNode.maxNumberOfCylinders,
                use_last_tracked_radius=self.ui.useLastTrackedRadius.checked,
                graph_branches=self.graph_branches,
                isNewBranch=self.ui.createBranch.text == "Create New Branch",
                progress_dialog=progress_dialog,
            )

            self.recenter3dView()

            # Select the starting markup node to ease future node placement
            slicer.app.applicationLogic().GetSelectionNode().SetActivePlaceNodeID(
                self.parameterNode.startingPoint.GetID()
            )

            self.checkCanStartRansac()
            self.checkCanStartSegmentation()

    def changeTextSize(self, value):
        """
        Update centerline label text size when moving the slider.
        """
        self.graph_branches.centerline_text_size = value
        for markup in self.graph_branches.centerline_markups:
            markup.GetDisplayNode().SetTextScale(value)

    def checkCanStartSegmentation(self, *args):
        """
        Enables the create segmentation button when at least one branch exist.
        """
        paintButton: qt.QPushButton = self.ui.paintButton
        paintButton.enabled = (
            len(self.graph_branches.centerline_markups) != 0
            and self.parameterNode.inputVolume
        )

        if self.segmentationNode is None or self.segmentationNode.GetScene() is None:
            self.segmentationNode = None
            if self.nodeDeletionObserverTag is not None:
                slicer.mrmlScene.RemoveObserver(self.nodeDeletionObserverTag)
                self.nodeDeletionObserverTag = None

    def onStartSegmentationButton(self) -> None:
        """
        Compute and create the segments inside the segmentation node according
        to the graph_branches architecture.
        """
        with slicer.util.tryWithErrorDisplay(
            "Failed to compute segmentation.", waitCursor=True
        ):
            # Get the currently selected segmentation node of the segment editor widget
            selected_segmentation = self.ui.SegmentEditorWidget.segmentationNode()

            if selected_segmentation is not None:
                self.segmentationNode = selected_segmentation
            elif (
                self.segmentationNode is None
                or self.segmentationNode.GetScene() is None
            ):
                self.segmentationNode = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLSegmentationNode"
                )
                self.segmentationNode.SetName("Segmentation")

            self.segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(
                self.parameterNode.inputVolume
            )
            self.segmentationNode.CreateDefaultDisplayNodes()

            # We do pause the tracking of segmentation deletion
            slicer.mrmlScene.RemoveObserver(self.nodeDeletionObserverTag)
            self.nodeDeletionObserverTag = None

            branch_draw_order = _compute_draw_order(
                self.graph_branches.nodes, self.graph_branches.edges
            )

            # Create the segments and paint them
            paint_segments(
                self.parameterNode.inputVolume,
                self.graph_branches.centerlines,
                self.graph_branches.names,
                self.graph_branches.centerline_radius,
                branch_draw_order,
                self.segmentationNode,
                self.parameterNode.reductionFactor,
                self.parameterNode.reductionThreshold,
                self.parameterNode.contourDistance,
                self.ui.mergeAllVessels.checked,
            )

            # Remove segmentation from the UI
            self.ui.SegmentEditorWidget.setSegmentationNode(None)
            # Remove volume node from the UI
            self.ui.SegmentEditorWidget.setSourceVolumeNode(None)

            # Set the current segmentation into the UI
            self.ui.SegmentEditorWidget.setSegmentationNode(self.segmentationNode)
            # Set the current volume into the UI
            self.ui.SegmentEditorWidget.setSourceVolumeNode(
                self.parameterNode.inputVolume
            )

            # Hide markup nodes
            for branch in (
                self.graph_branches.centerline_markups
                + self.graph_branches.contour_points_markups
            ):
                branch.GetDisplayNode().SetVisibility(False)

            for branch in self.graph_branches.tree_widget._branchDict.values():
                branch.setIcon(TreeColumnRole.VISIBILITY_CENTER, Icons.visibleOff)
                branch.setIcon(TreeColumnRole.VISIBILITY_CONTOUR, Icons.visibleOff)
            self.ui.showCenterlineButton.text = "Show Centerlines"
            self.ui.showContourPointsButton.text = "Show Contour Points"

            self.checkCanStartSegmentation()

            # We track segmentation deletion
            self.nodeDeletionObserverTag = slicer.mrmlScene.AddObserver(
                slicer.vtkMRMLScene.NodeAboutToBeRemovedEvent,
                self.checkCanStartSegmentation,
            )

    def onLockButton(self) -> None:
        """
        Lock / Unlock branches, disable any interactions you may have with branches.
        When locked, you cannot accidentally select / move points.
        """
        button = self.ui.lockButton
        markups = (
            self.graph_branches.centerline_markups
            + self.graph_branches.contour_points_markups
        )

        if button.checked:
            button.text = "Unlock Tree"
            for markup in markups:
                markup.LockedOn()
        else:
            button.text = "Lock Tree"
            for markup in markups:
                markup.LockedOff()

    def onLoadTreeArchitecture(self) -> None:
        """
        Loads a vessel tree architecture from a .JSON file.

        Ask first if the user wants to delete the current tree.
        If not, does nothing.
        """
        dialog = qt.QFileDialog()
        file_path = dialog.getOpenFileName(
            None, "Choose a file", "", "JSON file (*.json)"
        )

        # cancel any action if the user cancel / close the window / press escape
        if not file_path:
            return

        with slicer.util.tryWithErrorDisplay(
            "Failed to load tree architecture.", waitCursor=False
        ):
            with open(file_path) as f:
                js_graph = json.load(f)
            graph: nx.DiGraph = json_graph.node_link_graph(js_graph)

            # We ask to clear the tree before loading the new one, if not we do nothing
            if not self.graph_branches.clear_all():
                return
            self.checkCanStartSegmentation()
            self.checkCanStartRansac()

        self.graph_branches.load_branches_from_graph(graph)

        self.checkCanStartRansac()
        self.checkCanStartSegmentation()
        self.recenter3dView()


#
# pulmonary_arteries_segmentor_moduleLogic
#


class pulmonary_arteries_segmentor_moduleLogic(ScriptedLoadableModuleLogic):
    """
    This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        """
        Returns parent's parameter node.
        """
        return pulmonary_arteries_segmentor_moduleParameterNode(
            super().getParameterNode()
        )

    def processBranch(
        self,
        raw_volume: vtkMRMLScalarVolumeNode,
        starting_point_list: vtkMRMLMarkupsFiducialNode,
        direction_point_list: vtkMRMLMarkupsFiducialNode,
        percent_inlier_points: float,
        inlier_threshold: float,
        starting_radius: float,
        centerline_resolution: float,
        maximum_turn_angle: float,
        min_number_of_attempts: int,
        max_number_of_attempts: int,
        max_number_of_cylinders: int,
        use_last_tracked_radius: bool,
        graph_branches: GraphBranches,
        isNewBranch: bool,
        progress_dialog: CustomStatusDialog,
    ) -> GraphBranches:
        """
        Prepare the volume and run the RANSAC algorithm using the user parameters.

        Parameters
        ----------
        raw_volume: Input volume.
        starting_point_list: Slicer item containing the point defining the beginning of a cylinder (center of the bottom circle).
        direction_point_list: Slicer item containing the point defining the end of a cylinder (center of the top circle).
        percent_inlier_points: amount of point tagged as inlier in order to validate that a cylinder is correct.
        inlier_threshold: threshold from which a point is defined as an inlier of a cylinder or not.
        starting_radius: radius of the first cylinder from which the tracking starts
        centerline_resolution: maximum distance allowed between centerline points, later use to decide if we need to refine or not the centerline.
        maximum_turn_angle: maximum angle a cylinder can deviate from the last one fitted.
        min_number_of_attempts: the minimum number of attempts done to find a fitting cylinder.
        max_number_of_attempts: the maximum number of attempts to find a fitting cylinder.
        max_number_of_cylinders: the maximum number of cylinder tracked in one tracking.
        use_last_tracked_radius: flag to indicate whether we override the radius value entered with the radius
            of the closest cylinder of the input cylinder.
        graph_branches: object holding the graph of vessels branches.
        isNewBranch: flag to tell if it is the first branch or not.
        progress_dialog: UI window to inform the user on the state of the branch tracking.

        Returns
        ----------

        GraphBranches
        Updated graph
        """

        # Prepare the volume object
        vol = Volume.from_scalar_volume(raw_volume)

        # Prepare the starting and direction point objects
        starting_point = np.array([0, 0, 0])
        starting_point_list.GetNthControlPointPosition(
            starting_point_list.GetNumberOfControlPoints() - 1, starting_point
        )

        direction_point = np.array([0, 0, 0])
        direction_point_list.GetNthControlPointPosition(
            direction_point_list.GetNumberOfControlPoints() - 1, direction_point
        )

        # Convert degrees into radians
        radians_angle = radians(maximum_turn_angle)

        graph_branches = run_ransac(
            vol=vol,
            starting_point=starting_point,
            direction_point=direction_point,
            starting_radius=starting_radius,
            percent_inlier_points=percent_inlier_points,
            inlier_threshold=inlier_threshold,
            centerline_resolution=centerline_resolution,
            maximum_turn_angle=radians_angle,
            min_number_of_attempts=min_number_of_attempts,
            max_number_of_attempts=max_number_of_attempts,
            max_number_of_cylinders=max_number_of_cylinders,
            use_last_tracked_radius=use_last_tracked_radius,
            graph_branches=graph_branches,
            isNewBranch=isNewBranch,
            progress_dialog=progress_dialog,
        )

        return graph_branches


#
# pulmonary_arteries_segmentor_moduleTest
#


class pulmonary_arteries_segmentor_moduleTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def runTest(self):
        import unittest

        try:
            to_reload = [
                key for key in sys.modules.keys() if key.startswith("module_tests")
            ]
            to_reload.sort(key=len, reverse=True)
            # We reload this module first because the other test may depend on it
            for file_to_reload in to_reload:
                sys.modules[file_to_reload] = importlib.reload(
                    sys.modules[file_to_reload]
                )
        except Exception as e:
            print(f"Exception occurred while reloading\n{e}")

        from module_tests import (
            Branch_treeTest,
            CylinderTest,
            Cylinder_ransacTest,
            DependenciesInstallationTest,
            Graph_branchesTest,
            HelperTest,
            JitCompiledFunctionsTest,
            RansacTest,
            RegionGrowingSeedsTest,
            SegmentTest,
            VolumeTest,
        )

        testCases = [
            DependenciesInstallationTest,
            HelperTest,
            VolumeTest,
            SegmentTest,
            JitCompiledFunctionsTest,
            Branch_treeTest,
            CylinderTest,
            Cylinder_ransacTest,
            Graph_branchesTest,
            RegionGrowingSeedsTest,
            RansacTest,
        ]

        suite = unittest.TestSuite(
            [unittest.TestLoader().loadTestsFromTestCase(case) for case in testCases]
        )
        test_results = unittest.TextTestRunner(verbosity=3).run(suite)
        slicer.mrmlScene.Clear()

        summary = f"""
Total tests run: {test_results.testsRun}
Failures: {len(test_results.failures)}
Errors: {len(test_results.errors)}
Skipped: {len(getattr(test_results, 'skipped', []))}
Expected Failures: {len(getattr(test_results, 'expectedFailures', []))}
Unexpected Successes: {len(getattr(test_results, 'unexpectedSuccesses', []))}

Overall result: {"OK" if not test_results.failures and not test_results.errors else "FAILED, open the python console for more details."}
"""

        slicer.util.infoDisplay(text=summary, windowTitle="Tests results")
        slicer.app.processEvents()
