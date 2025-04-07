from typing import Union
import numpy as np
import math

from .jit_compiled_functions import (
    numba_close,
    numba_fit_cylinder_ransac,
    numba_filter_points,
    numba_fit_3_points_cylinder,
)

from .popup_utils import CustomStatusDialog

from .volume import Volume


from .helper import sample_gauss_sphere, sample_half_gauss_sphere, gradient_central_dif
from .cylinder import Cylinder


class Config:
    """
    Configuration for RANSAC algorithm
    """

    _min_radius = 0.5
    _max_radius = 5

    def __init__(
        self,
        nb_test_min=0,
        nb_test_max=1000,
        percent_inliers=0.5,
        threshold=0.1,
        radius_min=0.5,
        radius_max=1.5,
        angle_max=np.pi / 3,
        nb_cyl_dirs=81,
        nb_ray_dirs=162,
        n_samples=128,
        ray_length=2,
        nb_iter=1000,
    ):
        """
        Initialize algorithm's configuration

        Args:
            nb_test_min (int, optional): Min number of RANSAC tests. Clamped to [0,+infinity). Defaults to 0.
            nb_test_max (int, optional): Max number of RANSAC tests. Clamped to [0,+infinity). Defaults to 1000.
            percent_inliers (float, optional): Percentage of inliers to accept a candidate. Clamped to [0,1].
                                               Defaults to 0.5.
            threshold (float, optional): Proportion of the previous radius to take as distance threshold to elect
                                         inliers. Absolute value taken, so that it is positive. Defaults to 0.1.
            radius_min (float, optional): Proportion of the previous radius to take as min values of acceptable
                                          radius. Clamped to [_min_radius,_max_radius]. Defaults to 0.5.
            radius_max (float, optional): Proportion of the previous radius to take as max values of acceptable
                                          radius. Clamped to [_min_radius,_max_radius]. Defaults to 1.5.
            angle_max (float, optional): Maximum angle. Clamped to [0,Pi/2]. Defaults to np.pi/2.
            nb_cyl_dirs (int, optional): Number of directions to consider for the cylinder axis in RANSAC (clamped
                                         to 21 as a minimum, see helper.sample_half_gauss_sphere). Defaults to 81.
            nb_ray_dirs (int, optional): Number of directions to consider to cast rays from the vessel center
                                         (clamped to 42 as a minimum, see helper.sample_gauss_sphere).
                                         Defaults to 162.
            n_samples (int, optional): Number of samples to extract on each ray cast. Defaults to 128.
            ray_length (int, optional): Length of a cast ray, as a proportion of the previous radius. Defaults to 2.
            nb_iter (int, optional): Number of iterations for the algorithm. Defaults to 1000.
        """

        self.nb_test_min = nb_test_min
        self.nb_test_max = nb_test_max
        self.pct_inl = percent_inliers
        self.threshold = threshold
        self.r_min = radius_min
        self.r_max = radius_max
        self.a_max = angle_max
        self.cyl_dir_set = nb_cyl_dirs
        self.ray_dir_set = nb_ray_dirs
        self.n_samples = n_samples
        self.ray_len = ray_length
        self.nb_iter = nb_iter

        # To be done: proportion of the previous height to advance to get the new centerf
        self.advance_ratio = 0.5

    @property
    def nb_test_min(self):
        """
        Getter for min number of RANSAC tests

        Returns:
            int: Min number of RANSAC tests
        """

        return self._nb_test_min

    @nb_test_min.setter
    def nb_test_min(self, n):
        """
        Set nb_test_min to n
        Clamped to [0,+infinity).

        Args:
            n (int): New min number of RANSAC tests
        """

        self._nb_test_min = n if n > 0 else 0

    @property
    def nb_test_max(self):
        """
        Getter for max number of RANSAC tests

        Returns:
            int: Max number of RANSAC tests
        """

        return self._nb_test_max

    @nb_test_max.setter
    def nb_test_max(self, n):
        """
        Set nb_test_max to n
        Clamped to [0,+infinity).

        Args:
            n (int): New max number of RANSAC tests
        """

        self._nb_test_max = n if n > 0 else 0

    @property
    def pct_inl(self):
        """
        Getter for percentage of inliers to accept a candidate.

        Returns:
            float: Percentage of inliers to accept a candidate
        """

        return self._pct_inl

    @pct_inl.setter
    def pct_inl(self, p):
        """
        Set pct_inl to p, the minimum percentage of inliers to select a candidate cylinder. Clamped to [0,1]

        Args:
            p (float): New percentage of inliers to accept a candidate
        """

        if p < 0:
            p = 0
        elif p > 1:
            p = 1

        self._pct_inl = p

    @property
    def threshold(self):
        """
        Getter of proportion of the previous radius to take as distance threshold to elect inliers

        Returns:
            float: Proportion of the previous radius to take as distance threshold to elect inliers
        """

        return self._threshold

    @threshold.setter
    def threshold(self, t):
        """
        Sets threshold to t, the largest distance to the cylinder for a point to be selected as inlier, expressed as a
        proportion of the current radius.
        Take the absolute value

        Args:
            t (float): New proportion of the previous radius to take as distance threshold to elect inliers
        """

        self._threshold = math.fabs(t)

    @property
    def r_min(self):
        """
        Getter of proportion of the previous radius to take as min values of acceptable radius

        Returns:
            float: Proportion of the previous radius to take as min values of acceptable radius
        """

        return self._r_min

    @r_min.setter
    def r_min(self, r):
        """
        Set r_min to r, the proportion of the smallest acceptable candidate radius to the current one.
        Clamped to [_min_radius,_max_radius]
        Also set r_max to r if r_max < r

        Args:
            r (float): Proportion of the smallest acceptable candidate radius
        """

        if r < Config._min_radius:
            r = Config._min_radius
        elif r > Config._max_radius:
            r = Config._max_radius

        self._r_min = r

        if not hasattr(self, "_r_max") or self.r_max < r:
            self.r_max = r

    @property
    def r_max(self):
        """
        Getter of proportion of the previous radius to take as max values of acceptable radius

        Returns:
            float: Proportion of the previous radius to take as max values of acceptable radius
        """

        return self._r_max

    @r_max.setter
    def r_max(self, r):
        """
        Set r_max to r, the proportion of the largest acceptable candidate radius to the current one
        Clamped to [_min_radius,_max_radius]
        Also set r_min to r if r_min > r, and set ray_len to r if ray_len < r

        Args:
            r (float): Proportion of the largest acceptable candidate radius to the current one
        """

        if r < Config._min_radius:
            r = Config._min_radius
        elif r > Config._max_radius:
            r = Config._max_radius

        self._r_max = r

        if not hasattr(self, "_r_min") or self.r_min > r:
            self.r_min = r

        if not hasattr(self, "_ray_len") or self.ray_len < r:
            self.ray_len = r

    @property
    def a_max(self):
        """
        Getter of maximum angle

        Returns:
            float: Maximum angle
        """

        return self._a_max

    @a_max.setter
    def a_max(self, a):
        """
        Set a_max to a, the maximum angle between the candidate direction and the current one
        Clamped to [0,Pi/2].
        Setting a_max to anything larger that pi consists in accepting any direction. Therefore, a_max is set to pi
        in such cases

        Args:
            a (float): New maximum angle
        """

        if a < 0:
            a = 0
        elif a > np.pi / 2:
            a = np.pi / 2

        self._a_max = a

    @property
    def cyl_dir_set(self):
        """
        Getter of directions to consider for the cylinder axis in RANSAC

        Returns:
            np.array(dtype=np.float64): Directions to consider for the cylinder axis in RANSAC
        """

        return self._cyl_dirs

    @cyl_dir_set.setter
    def cyl_dir_set(self, nb_cyl_dirs):
        """
        Setter of directions to consider for the cylinder axis in RANSAC

        Args:
            nb_cyl_dirs (int): Number of directions to consider for the cylinder axis in RANSAC
        """

        # Only update _cyl_dirs if necessary
        if not hasattr(self, "_cyl_dirs") or self._cyl_dirs.shape[0] < nb_cyl_dirs:
            # Regular sampling of the half Gaussian sphere
            self._cyl_dirs = sample_half_gauss_sphere(nb_cyl_dirs)

    @property
    def nb_cyl_dirs(self):
        """
        Getter of number of directions to consider for the cylinder axis in RANSAC

        Returns:
            int: Number of directions to consider for the cylinder axis in RANSAC
        """

        return self.cyl_dir_set.shape[0]

    @property
    def ray_dir_set(self):
        """
        Getter of directions to consider to cast rays from the vessel center

        Returns:
            np.array(dtype=np.float64): Directions to consider to cast rays from the vessel center
        """

        return self._ray_dirs

    @ray_dir_set.setter
    def ray_dir_set(self, nb_ray_dirs):
        """
        Setter of directions to consider to cast rays from the vessel center

        Args:
            nb_ray_dirs (int): Number of directions to consider to cast rays from the vessel center
        """

        # Only update _ray_dirs if necessary
        if not hasattr(self, "_ray_dirs") or self._ray_dirs.shape[0] < nb_ray_dirs:
            # Regular sampling of the half Gaussian sphere
            self._ray_dirs = sample_gauss_sphere(nb_ray_dirs)

    @property
    def nb_ray_dirs(self):
        """
        Getter of number of directions to consider to cast rays from the vessel center

        Returns:
            int: Number of directions to consider to cast rays from the vessel center
        """

        return self.ray_dir_set.shape[0]

    @property
    def n_samples(self):
        """
        Getter of number of samples to extract on each ray cast

        Returns:
            int: Number of samples to extract on each ray cast
        """

        return self._n_samples

    @n_samples.setter
    def n_samples(self, ns):
        """
        Setter of number of samples to extract on each ray cast

        Args:
            ns (int): New number of samples to extract on each ray cast
        """

        if ns <= 0:
            self._n_samples = 1
        else:
            self._n_samples = ns

    @property
    def ray_len(self):
        """
        Getter of length of a cast ray

        Returns:
            int: Length of a cast ray
        """

        return self._ray_len

    @ray_len.setter
    def ray_len(self, rl):
        """
        Sets the length of a cast ray, as a proportion of the current radius estimate
        Must be larger than self.r_max.
        If not, then self.ray_len is set to self.r_max, discarding rl

        Args:
            rl (int): The value to set. If below r_max, it is discarded and ray_len is set to r_max
        """

        if rl <= self.r_max:
            self._ray_len = self.r_max
        else:
            self._ray_len = rl

    @property
    def nb_iter(self):
        """
        Getter of number of iterations for the algorithm

        Returns:
            int: Number of iterations for the algorithm
        """

        return self._nb_iter

    @nb_iter.setter
    def nb_iter(self, n):
        """
        Setter of number of iterations for the algorithm

        Args:
            n (int): New number of iterations for the algorithm
        """

        self._nb_iter = n if n > 0 else 0


def sample(vol: Volume, center, radius, n_samples, dirs):
    """
    Cast rays in a volume from center along each direction in dirs with length radius.
    Retrieves the point of minimum directional gradient along each ray.

    Args:
        vol (volume): Input volume
        center (np.array(dtype=np.float64)): A 3-vector, start of each ray
        radius (float): The length of each cast ray
        n_samples (int): Number of samples to extract along each ray
        dirs (np.array(dtype=np.float64)): The directions to cast rays towards, as a Nx3 array. Each direction has
                                           to be normalized.

    Returns:
        np.array(dtype=np.float64): One point along each ray, all assembled in a Nx3 array.
    """

    p = np.empty((dirs.shape[0], 3))

    for i, d in enumerate(dirs):
        start = center
        end = center + d * radius
        interpolated_coords, c = vol.get_line(start, end, n_samples)

        # Exclude first and last points whose gradient is invalid
        k = np.argmin(gradient_central_dif(interpolated_coords)[1:-1]) + 1
        p[i] = c[k]

    return p


def sample_around_cylinder(
    vol: Volume, cyl: Cylinder, cfg: Config, max_tries: int = 5
) -> Union[tuple[None, None], tuple[np.ndarray, float]]:
    """
    Compute the current cylinder inliers without moving the cylinder center.
    For more details about the process involved, please refer to the next_cylinder function,
    since this function is a cheaper version of it.

    Args:
        vol (volume): Input volume
        cyl (cylinder): Current cylinder
        cfg (config): Tracking configuration

    Returns:
        np.array(dtype=np.float64): Current cylinder's inlier point set
    """

    cyl_to_refine = cyl.copy()
    ray_len = cyl_to_refine.radius * cfg.ray_len
    err_threshold = cyl_to_refine.radius * cfg.threshold

    for _ in range(max_tries):
        p = sample(vol, cyl_to_refine.center, ray_len, cfg.n_samples, cfg.ray_dir_set)
        p = numba_filter_points(p, cyl_to_refine.center, 0.1 * ray_len, 0.9 * ray_len)
        i_max = cyl_to_refine.select_inliers(p, err_threshold)

        if i_max.shape[0] < 3:
            continue

        cyl_to_refine.fix_height(i_max)
        if cyl_to_refine.radius > 4:
            cyl_to_refine.height = cyl_to_refine.radius

        cyl_to_refine.contour_points = i_max
        return cyl_to_refine
    return None


def next_cylinder(vol: Volume, cyl: Cylinder, cfg: Config):
    """
    Compute next cylinder

    Args:
        vol (volume): Input volume
        cyl (cylinder): Current cylinder
        cfg (config): Tracking configuration

    Returns:
        cylinder: Next cylinder
        np.array(dtype=np.float64): Next cylinder's inlier point set
    """
    # We try with original cfg.advance_ratio, and if it fails, try again with 2*cfg.advance_ratio (typical case when
    # the artery is highly curved)
    for li in [1, 2]:
        # Compute guess for next cylinder center: advance from current one by cfg.advance_ratio times its height along
        # direction li is here in case the original advance_ratio (0.5 by default) was not large enough to ensure
        # sufficient progress in the tracking
        next_center = cyl.center + li * cfg.advance_ratio * cyl.height * cyl.direction

        # Sets some parameters from cfg
        r_min = cyl.radius * cfg.r_min
        r_max = cyl.radius * cfg.r_max
        ray_len = cyl.radius * cfg.ray_len
        err_threshold = cyl.radius * cfg.threshold

        # Extract points. Remove (filter out) points at both extremities because the 3rd order interpolation might
        # provide tainted gradient values
        p = sample(vol, next_center, ray_len, cfg.n_samples, cfg.ray_dir_set)
        p = numba_filter_points(p, next_center, 0.1 * ray_len, 0.9 * ray_len)

        # Stop if less than 3 points remain after filtering
        if p.shape[0] < 3:
            return None

        # Sort cylinder directions starting with the one most aligned with cyl.direction
        idx = np.argsort(-np.abs(cyl.direction @ cfg.cyl_dir_set.T))

        # Look for best cylinder:
        #   - review axes in idx-based order
        #   - as soon as a cylinder is found with more than cfg.pct_inl inlier rate, keep it and stop searching -
        #     otherwise list cylinders with an inlier rate larger than cfg.pct_inl/2:
        #       the cylinder with the best inlier rate will be kept, if any
        p_max = 0
        c_max = Cylinder()
        i_max = np.empty((0, 3))

        for axis in cfg.cyl_dir_set[idx]:
            if np.abs(cyl.direction @ axis) < np.cos(cfg.a_max):
                continue

            c_basis, inliers, p_inl = numba_fit_cylinder_ransac(
                p,
                axis,
                cfg.nb_test_min,
                cfg.nb_test_max,
                cfg.pct_inl,
                r_min,
                r_max,
                err_threshold,
            )
            if numba_close(p_inl, 0.0):
                return None

            c = Cylinder(
                *numba_fit_3_points_cylinder(c_basis[0], c_basis[1], c_basis[2], axis)
            )

            if p_inl > cfg.pct_inl:
                p_max = p_inl
                c_max = c
                i_max = inliers
                break

            # Also list cylinder with inlier rate > cfg.pct_inl/2
            elif p_inl > cfg.pct_inl / 2 and p_max < p_inl:
                # Update cyl with max inliers
                p_max = p_inl
                c_max = c
                i_max = inliers

        # Need at least 3 points to fit a cylinder.
        # Else: stop, no valid cylinder was found
        if i_max.shape[0] < 3:
            return None

        # Now refine the cylinder that was found
        r_min = c_max.radius * cfg.r_min
        r_max = c_max.radius * cfg.r_max
        ray_len = c_max.radius * cfg.ray_len
        err_threshold = c_max.radius * cfg.threshold

        # Update c_max, with same direction, but recompute points and other parameters
        # Update center at a median position along axis wrt inliers.
        c_max.fix_center(i_max)

        # Re-extract and filter points from this new center
        p = sample(vol, c_max.center, ray_len, cfg.n_samples, cfg.ray_dir_set)
        p = numba_filter_points(p, c_max.center, 0.1 * ray_len, 0.9 * ray_len)

        # Select the inliers
        i_max = c_max.select_inliers(p, err_threshold)

        # Need at least 3 points to fit a cylinder. Else: stop
        if i_max.shape[0] < 3:
            return None

        # Re-update center wrt new inliers
        c_max.fix_center(i_max)

        # Compute height and keep inscribed inlier points
        i_max = c_max.fix_height(i_max)

        # Refine cylinder parameters, in particular to smooth its direction that was originally picked with a discrete
        # set
        c_max.refine(i_max)

        # Update inlier set according to new cylinder
        i_max = c_max.select_inliers(p, err_threshold)

        # Need at least 3 points to fit a cylinder. Else: stop
        if i_max.shape[0] < 3:
            return None

        # Re-update center wrt inliers
        c_max.fix_center(i_max)

        # Fixes cylinder direction so that it points in the same direction as the tracking advance necessary for next
        # step (computation of next guess)
        if c_max.direction @ cyl.direction < 0:
            c_max.direction = -c_max.direction

        # Fixes height, and select inliers with this height
        i_max = c_max.fix_height(i_max)

        # Test if a sufficient advance was made.
        # If not, we might try with a double advance_ratio (initial loop over li)
        if c_max.radius > 4:
            c_max.height = c_max.radius
        dist_centers = np.linalg.norm(c_max.center - cyl.center)
        if dist_centers >= cyl.height / 2 and (
            cyl.height == 0 or dist_centers <= cyl.height * 2
        ):
            c_max.contour_points = i_max
            return c_max

    # No new cylinder was found
    return None


def track_cylinder(vol, cyl, cfg):
    """
    Generator that tracks cylinders in a volume, starting with an initial guess

    Args:
        vol (volume): Input volume
        cyl (cylinder): Initial cylinder guess
        cfg (config): Configuration for the tracking

    Yields:
        cylinder: Next cylinder
        np.array(dtype=np.float64): Next cylinder's inlier set
    """

    for _ in range(cfg.nb_iter):
        c_max = next_cylinder(vol, cyl, cfg)

        if c_max is not None:
            yield c_max
            cyl = c_max.copy()
        else:
            break


def track_branch(
    vol: Volume,
    cyl: Cylinder,
    cfg: Config,
    already_tracked_cylinders: list[Cylinder],
    progress_dialog: CustomStatusDialog,
) -> list[Cylinder]:
    """
    Performs the tracking in a volume, given an input cylinder and a configuration

    Args:
        vol (Volume): Input volume.
        cyl (cylinder): Input cylinder from which the tracking starts.
        cfg (config): Configuration for the tracking.
        centerline (np.ndarray): Center of the tracked cylinders defining a centerline.
        centerline_radius (list[float]): list containing the radius of each of the cylinders tracked.
        contour_point (list[list[np.ndarray]]): Contour points of each cylinder tracked (each centerline point has a certain amount of contour point).
        already_tracked_cylinders (list[cylinder.Cylinder]): already tracked cylinders, necessary to stop the algorithm if the tracking
            goes backward.
        progress_dialog (CustomStatusDialog): UI window to inform the user on the state of the branch tracking.
    """

    contour_points_cpt = 0
    current_branch_cylinders = []

    for tracked_cylinder in track_cylinder(vol, cyl, cfg):
        # Criteria for acceptance: Need to be better justified especially third one
        #   1- Valid cylinder (i.shape[0] > 0)
        #   2- Sufficient advance: |c.center-cyl.center| > cyl.height/4
        #   3- Not redundant: d(c,branch) > cyl.radius/10
        # (Note: what if cyl.height/4 < cyl.radius/10?)

        if (
            tracked_cylinder.contour_points.shape[0] > 0
            and not tracked_cylinder.is_redundant(already_tracked_cylinders)
            and not tracked_cylinder.is_redundant(current_branch_cylinders)
        ):
            current_branch_cylinders.append(tracked_cylinder)

            contour_points_cpt += len(tracked_cylinder.contour_points)

            progress_dialog.set_text(
                f"Centerline points found: {len(current_branch_cylinders)}\nContour points found: {contour_points_cpt}"
            )

        else:
            break

    progress_dialog.close()
    return current_branch_cylinders
