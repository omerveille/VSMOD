import numpy as np
import scipy.ndimage as sndi
from . import helper
from slicer import vtkMRMLScalarVolumeNode
from scipy.ndimage import spline_filter
import slicer
import vtk


class Volume:
    """
    Class to represent a volume
    """

    def __init__(self, data=np.zeros((0, 0, 0)), ijk_to_ras=np.eye(4), order=3):
        """
        Initializes a volume

        Args:
            data (np.array(dtype=np.float64), optional): Volume's data. Defaults to np.zeros((0, 0, 0)).
            ijk_to_ras (np.array(dtype=np.float64), optional): Volume's IJK to RAS transformation.
                                                               Defaults to np.eye(4).
            order int: Interpolation order

        Raises:
            ValueError: If volume's data or matrix transformation isn't good shape
        """

        if len(data.shape) != 3 or ijk_to_ras.shape != (4, 4):
            raise ValueError

        data = spline_filter(data, order=order)

        self._vol, self.ijk_to_ras = data, ijk_to_ras

        # Default to linear interpolation
        self._order = order

    @classmethod
    def from_scalar_volume(cls, volume: vtkMRMLScalarVolumeNode):
        """
        Initializes a volume

        Args:
            volume (vtkMRMLScalarVolumeNode): Volume node.

        Raises:
            TypeError: If the volume's argument type is not correct
        """
        if not isinstance(volume, vtkMRMLScalarVolumeNode):
            raise TypeError(
                f"Expected volume to be a vtkMRMLScalarVolumeNode got a {type(volume)}"
            )

        interpolation_order = 3
        data = slicer.util.array(volume.GetID())
        data = data.swapaxes(0, 2)
        data = spline_filter(data, order=interpolation_order)

        ijk_to_ras = vtk.vtkMatrix4x4()
        volume.GetIJKToRASMatrix(ijk_to_ras)
        np_ijk_to_ras = np.zeros(shape=(4, 4))
        ijk_to_ras.DeepCopy(np_ijk_to_ras.ravel(), ijk_to_ras)
        return Volume(data=data, ijk_to_ras=np_ijk_to_ras, order=interpolation_order)

    def __call__(self, p):
        """
        Compute the values at positions p by interpolation.
        See order property for the interpolation (scipy.ndimages.map_coordinates is used)

        Args:
            p (np.array(dtype=np.float64)): 3D points in RAS coordinates

        Returns:
            np.array(dtype=np.float64): Volume's data mapped to new coordinates
        """

        return sndi.map_coordinates(
            self._vol, self.transf_ras_to_ijk(p).T, order=self.order, prefilter=False
        )

    def transf_ijk_to_ras(self, p):
        """
        Transforms a set of IJK coordinates into RAS coordinates

        Args:
            p (np.array(dtype=np.float64)): Points in IJK coordinates

        Returns:
            np.array(dtype=np.float64): Points in RAS coordinates
        """

        return helper.homogenize(p) @ self._ijk_to_ras.T[:, :3]

    def transf_ras_to_ijk(self, p):
        """
        Transforms a set of RAS coordinates into IJK coordinates

        Args:
            p (np.array(dtype=np.float64)): Points in RAS coordinates

        Returns:
            np.array(dtype=np.float64): Points in IJK coordinates
        """

        return helper.homogenize(p) @ self._ras_to_ijk.T[:, :3]

    def get_line(self, start, end, n_samples=128):
        """
        Extract n_samples values along a line going from start to end (both points are included). Points are expressed
        in RAS coordinates. Their coordinates can be retrieved by a simple call to np.linspace(start,end,n_samples).
        See also the order property to tune the order of the spline for the interpolation
        It defaults to one for a linear interpolation

        Args:
            start (np.array(dtype=np.float64)): Samples start position
            end (np.array(dtype=np.float64)): Samples end position
            n_samples (int, optional): Number of samples. Defaults to 128.

        Returns:
            np.array(dtype=np.float64): Interpolated coordinates values
            np.array(dtype=np.float64): Source coordinate values
        """

        coord = np.linspace(start, end, n_samples)

        return self(coord), coord

    @property
    def ijk_to_ras(self):
        """
        Getter for IJK to RAS transform

        Returns:
            np.array(dtype=np.float64): IJK to RAS transform
        """

        return self._ijk_to_ras

    @ijk_to_ras.setter
    def ijk_to_ras(self, m):
        """
        Setter for IJK to RAS transform.
        Modify the other transforms too.

        Args:
            m (np.array(dtype=np.float64)): New IJK to RAS transform
        """

        self._ijk_to_ras = m.copy()
        self._ras_to_ijk = np.linalg.inv(self._ijk_to_ras)

    @property
    def ras_to_ijk(self):
        """
        Getter for RAS to IJK transform

        Returns:
            np.array(dtype=np.float64): RAS to IJK transform
        """

        return self._ras_to_ijk

    @ras_to_ijk.setter
    def ras_to_ijk(self, m):
        """
        Setter for RAS to IJK transform.
        Modify the other transforms too.

        Args:
            m (np.array(dtype=np.float64)): New RAS to IJK transform
        """

        self._ras_to_ijk = m.copy()
        self._ijk_to_ras = np.linalg.inv(self._ras_to_ijk)

    @property
    def order(self):
        """
        Getter for order

        Returns:
            int: Order
        """

        return self._order

    @order.setter
    def order(self, value):
        """
        Setter for order

        Args:
            value (int): New order value

        Raises:
            ValueError: Negative order value
        """

        if value < 0:
            raise ValueError("Volume.order value must be positive or zero")

        self._order = value
