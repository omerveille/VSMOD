import numpy as np
import math
import scipy.optimize as scopt
from math import isclose


from .segment import Segment


class Cylinder:
    """
    Class to represent a cylinder
    """

    def __init__(
        self,
        center: np.ndarray = np.array([0, 0, 0], dtype=np.float64),
        radius: float = 1,
        direction: np.ndarray = np.array([0, 0, 1], dtype=np.float64),
        height: float = -1,
        contour_points: np.ndarray = np.empty(shape=(0, 3), dtype=np.float64),
    ):
        """
        Generate a new instance of cylinder.
        The radius should be positive (ValueError raised if negative or zero, see radius property setter)
        Direction is ensured to be of norm 1: ValueError raised if a zero-length vector is provided as input
        (see direction property setter)
        A negative height is always translated as -1 and encodes a cylinder of infinite length.
        Zero-height cylinders are handled

        Args:
            center (np.array(dtype=np.float64), optional): Cylinder's center position.
                                                           Defaults to np.array([0, 0, 0], dtype=np.float64).
            radius (int, optional): Cylinder's radius. Defaults to 1.
            direction (np.array(dtype=np.float64), optional): Cylinder's direction.
                                                              Defaults to np.array([0, 0, 1], dtype=np.float64).
            height (int, optional): Cylinder's height. Defaults to -1.
        """

        self.center: np.ndarray = center
        self.radius: float = radius
        self.direction: np.ndarray = direction
        self.height: float = height
        self.contour_points: np.ndarray = contour_points

    def __eq__(self, other_cyl: object) -> bool:
        if type(self) is not type(other_cyl):
            return False

        return (
            np.allclose(self.center, other_cyl.center)
            and isclose(self.radius, other_cyl.radius)
            and np.allclose(self.direction, other_cyl.direction)
            and isclose(self.height, other_cyl.height)
            and np.allclose(self.contour_points, other_cyl.contour_points)
        )

    def __str__(self) -> str:
        return f"Cylinder= center:{self.center.tolist()}|radius: {self.radius}|direction:{self.direction.tolist()}| height:{self.height}| contour_points:{self.contour_points}"

    def copy(self):
        """
        Copy cylinder

        Returns:
            cylinder: Copy of current cylinder
        """

        return Cylinder(
            self.center, self.radius, self.direction, self.height, self.contour_points
        )

    @property
    def center(self):
        """
        Getter of cylinder's center

        Returns:
            np.array(dtype=np.float64): Cylinder's center
        """

        return self._center

    @center.setter
    def center(self, c):
        """
        Setter of cylinder's center

        Args:
            c (np.array(dtype=np.float64)): New cylinder's center
        """

        self._center = np.asarray(c, dtype=np.float64)

    @property
    def radius(self):
        """
        Getter of cylinder's radius

        Returns:
            int: Cylinder's radius
        """

        return self._radius

    @radius.setter
    def radius(self, r):
        """
        Setter of cylinder's radius

        Args:
            r (int): New cylinder's radius

        Raises:
            ValueError: r <= 0
        """

        if r <= 0:
            raise ValueError("A cylinder's radius cannot have a negative length value.")
        else:
            self._radius = r

    @property
    def direction(self):
        """
        Getter of cylinder's direction

        Returns:
            np.array(dtype=np.float64): Cylinder's direction
        """

        return self._direction

    @direction.setter
    def direction(self, d):
        """
        Setter of cylinder's direction

        Args:
            d (np.array(dtype=np.float64)): New cylinder's direction

        Raises:
            ValueError: norm(d) == 0
        """

        d = np.asarray(d, dtype=np.float64)
        n = np.linalg.norm(d)

        if math.isclose(n, 0):
            raise ValueError(
                "The length of the direction vector of the cylinder is too close to 0.\nYou should put a bit more space between the starting and the direction point."
            )
        else:
            self._direction = d / n

    @property
    def height(self):
        """
        Getter of cylinder's height

        Returns:
            int: Cylinder's height
        """

        return self._height

    @height.setter
    def height(self, h):
        """
        Setter of cylinder's height
        Sets height to -1 (aka infinite length) if h<0

        Args:
            h (int): New cylinder's height
        """

        if h < 0:
            self._height = -1
        else:
            self._height = h

    @property
    def contour_points(self):
        """
        Getter of cylinder's contour_points

        Returns:
            np.ndarray: Cylinder's contour points
        """
        return self._contour_points

    @contour_points.setter
    def contour_points(self, contour_points):
        """
        Setter of cylinder's contour_points

        Args:
            contour_points (np.array(dtype=np.float64)): New cylinder's contour_points

        Raises:
            ValueError: shape != (X, 3)
        """

        contour_points = np.asarray(contour_points)
        if len(contour_points.shape) != 2 or contour_points.shape[1] != 3:
            raise ValueError(
                f"Wrong dimension for the contour points of the cylinder.\nExpected (X, 3) got {contour_points.shape}."
            )

        self._contour_points = contour_points

    def signed_distance(self, p):
        """
        Signed distance to a set of points p. Positive outside the cylinder.

        Args:
            p (np.array(dtype=np.float64)): Nx3 array containing set of points

        Returns:
            int: Distance between cylinder's center and the set of points
        """
        return self.distance(p) - self.radius

    def distance(self, p):
        """
        Distance to a set of points p.

        Args:
            p (np.array(dtype=np.float64)): Nx3 array containing set of points

        Returns:
            int: Distance between cylinder's center and the set of points
        """
        if len(p.shape) == 1:
            pa = p.reshape((1, 3))
        else:
            pa = p

        # Compute distance to segment of length self.height
        if self.height >= 0:
            d = self.height / 2 * self.direction
            dist_to_axis = Segment(self.center - d, self.center + d).distance_sqr(pa)

        # Distance to infinite line
        else:
            d = pa - self.center
            li = d @ self.direction
            dist_to_axis = np.sum(d * d, axis=1) - li * li

        # Handle case of numerical errors causing dist_to_axis to be negative
        dist_to_axis[dist_to_axis < 0] = 0
        dist = np.sqrt(dist_to_axis)

        if len(p.shape) == 1:
            return dist[0]

        return dist

    def select_inliers(self, p, threshold):
        """
        Compute inliers from a point set p, that lies with threshold distance to the cylinder

        Args:
            p (np.array(dtype=np.float64)): Input point set Nx3
            threshold (float): Max distance to the cylinder (absolute value)

        Returns:
            (np.array(dtype=np.float64)): Inlier points selected
        """

        d = np.fabs(self.signed_distance(p))
        i = np.where(d < threshold)

        return p[i]

    def fix_center(self, inliers):
        """
        Fixes the center location so that it lies at a median position along the axis, with respect to the inlier points

        Args:
            inliers (np.array(dtype=np.float64)): Inlier points
        """

        li = np.sort(self.direction @ (inliers - self.center).T)
        self.center += li[len(li) // 2] * self.direction

    def fix_height(self, inliers):
        """
        Fixes the height of the cylinder.
        First, only 75% of the inlier points are kept: those that are closest to the center, as per the distance along
        the cylinder axis
        Then, compute the height so that it encompasses all the inlier points.

        Args:
            inliers (np.array(dtype=np.float64)): Input inlier point set

        Returns:
            np.array(dtype=np.float64): The inlier points that are kept according to the height
        """

        # Compute distances to center along direction
        li = np.abs(self.direction @ (inliers - self.center).T)
        rm = len(li) // 4

        # Sort and removes the rm farthest points, use - for decreasing order
        idx = np.argsort(-li)[rm:]

        # Height is twice of the greatest distance of kept points
        self.height = 2 * li[idx[0]]

        # Keep the closest points
        return inliers[idx]

    def refine(self, inliers):
        """
        Refines the cylinder axis so that the distance to the inlier points is minimized

        Args:
            inliers (np.array(dtype=np.float64)): Inlier point set to consider
        """

        def cyl_to_param(cyl):
            return np.append(cyl.center, cyl.radius * cyl.direction)

        def param_to_cyl(param):
            return Cylinder(
                param[:3], np.linalg.norm(param[3:]), param[3:], self.height
            )

        def residue(param):
            cyl = param_to_cyl(param)
            d = cyl.signed_distance(inliers)

            return d @ d / inliers.shape[0]

        prm = cyl_to_param(self)
        res = scopt.minimize(residue, prm)

        if res.success:
            c = param_to_cyl(res.x)
            self.center = c.center
            self.radius = c.radius
            self.direction = c.direction

    def is_redundant(self, b):
        """
        Determine if the cylinder is redundant with a branch.

        Args:
            b (list): The branch, as a list of cylinders, ordered to form a polygonal line (list of joint segments)

        Returns:
            bool: True if self is redundant, meaning that it lies with self.radius/10 of a segment in the branch,
                  else False
                  False if there are less than 2 points (strictly) in the branch
        """

        r_max2 = (self.radius / 10) ** 2

        return (
            len(b) >= 2
            and min(
                [
                    Segment(s.center, e.center).distance_sqr(self.center)
                    for s, e in zip(b[:-1], b[1:])
                ]
            )
            < r_max2
        )


def dist_to_branch(p, b):
    """
    Find i such that [b[i].center,b[i+1].center] is the closest segment to p.

    Args:
        p (np.array(dtype=np.float64)): The query point
        b (list): A branch, as a list of cylinders

    Returns:
        np.array(dtype=np.float64): Minimum square distance
        int: Index of the closest cylinder with the minimum square distance
    """

    if len(b) == 0:
        return -1

    if len(b) == 1:
        d = np.asarray(p, dtype=float) - b[0].center
        return np.dot(d, d), 0

    cp = b[0].center
    cn = b[1].center

    d_min = Segment(cn, cp).distance_sqr(p)
    closest = 0

    for i, cyl_n in enumerate(b[2:]):
        cp = cn
        cn = cyl_n.center
        d = Segment(cn, cp).distance_sqr(p)

        if d < d_min:
            d_min = d

            # Starts at i=0 for cyl_n of index 2
            closest = i + 1

    return d_min, closest


def closest_branch(query_point, branch_array):
    """
    Returns the closest branch to point p, within a list of branches ba

    Args:
        query_point (np.array(dtype=np.float64)): The query point
        branch_array (list): An array of branches, which itself is an array of cylinders

    Returns:
        np.array(dtype=np.float64): Minimum square distance
        list: Closest branch
        int: Index to the closest branch
        int: Index returned by dist_to_branch
    """

    if len(branch_array) == 0:
        return []

    d_min, closest_cyl_idx = dist_to_branch(query_point, branch_array[0])
    best_branch = branch_array[0]
    idx_best_branch = 0
    idx_current_branch = 0

    for current_branch in branch_array[1:]:
        idx_current_branch += 1
        d, current_closest_cyl_idx = dist_to_branch(query_point, current_branch)

        if d < d_min:
            best_branch = current_branch
            idx_best_branch = idx_current_branch
            d_min = d
            closest_cyl_idx = current_closest_cyl_idx

    return d_min, best_branch, idx_best_branch, closest_cyl_idx
