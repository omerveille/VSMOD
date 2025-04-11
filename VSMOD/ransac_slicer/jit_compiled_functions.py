from numba.core.typing.templates import signature
from numba.extending import intrinsic
import numpy as np
from numba import njit, prange, types, config

"""
Make sure to put all the JIT compiled functions in the same file, cache invalidation will fail in the case of
nested JIT compiled functions in different files. (C.F https://numba.readthedocs.io/en/stable/user/jit.html#compilation-options --> caching)
"""

# Setup the threading layer for parallel parts of the code
config.THREADING_LAYER = "omp"


@njit(cache=True)
def numba_random_indices(n: int) -> np.ndarray:
    """
    Return 3 random index value and make sure they are distinct in a numba friendly way.

    Args:
        n (int): max index possible

        Returns:
            indexes (np.array(int)) : cylinder basis indexes
    """
    while True:
        indexes = np.random.randint(0, n, 3)
        if (
            indexes[0] != indexes[1]
            and indexes[1] != indexes[2]
            and indexes[0] != indexes[2]
        ):
            return indexes


@intrinsic(cache=True)
def _atomic_xchg(typingctx, ptr, val):
    """
    Implementation of an atomic exchange.
    Atomically swap a value pointed and returns it.
    More info at https://numba.readthedocs.io/en/stable/extending/high-level.html#implementing-intrinsics.

    Args:
        typingctx : the context
        ptr (int64*): the pointer to the value to be exchanged
        val (int64): the value that will replace the pointed value

        Returns:
            old (int) : the value pointed by the pointer
    """
    sig = signature(types.int64, types.CPointer(types.int64), types.int64)

    def codegen(context, builder, signature, args):
        ptr, val = args

        old = builder.atomic_rmw("xchg", ptr, val, "monotonic")
        return old

    return sig, codegen


@njit
def atomic_exchange(arr: np.ndarray, new_value: int) -> int:
    """
    Implementation of an atomic exchange.
    Atomically swap a value pointed and returns it.

    Args:
        arr (np.array(dtype=np.int64)): the single value array pointing to the value to be exchanged
        val (int64) : the value that will replace the pointed value

        Returns:
            old (int64) : the value pointed by the array's pointer
    """
    # Convert the array into an array of type CPointer
    ptr = arr.ctypes
    old = _atomic_xchg(ptr, new_value)
    return old


@njit(cache=True)
def numba_cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Custom cross product to counter poor performance of numpy cross for single vectors

    Args:
        a (np.array(dtype=np.float64)): 3-arrays
        b (np.array(dtype=np.float64)): 3-arrays

    Returns:
        np.array(dtype=np.float64): 3-arrays result
    """

    return np.array(
        [
            a[1] * b[2] - b[1] * a[2],
            b[0] * a[2] - a[0] * b[2],
            a[0] * b[1] - b[0] * a[1],
        ]
    )


@njit(cache=True)
def numba_fit_3_points_cylinder(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, direction: np.ndarray
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Determine the cylinder going through 3 points p0,p1 and p2 and whose axis is along direction.
    Note that if direction were not given, 5 points would be required.

    Args:
        p0 (np.array(dtype=np.float64)): First point to use
        p1 (np.array(dtype=np.float64)): Second point to use
        p2 (np.array(dtype=np.float64)): Third point to use
        direction (np.array(dtype=np.float64)): Direction to have cylinder axis

    Returns:
        center, radius, direction (tuple[np.ndarray, float, np.ndarray]): Respectively
        the center, the radius and the direction of the cylinder defined with 3 points
    """

    try:
        # Normalize direction (if possible)
        n = np.linalg.norm(direction)
        if numba_close(n, 0.0):
            return np.zeros(3, dtype=np.float64), -1.0, np.zeros(3, dtype=np.float64)

        direction /= n

        # Remove the component along direction
        q0 = p0 - (p0 @ direction) * direction
        q1 = p1 - (p1 @ direction) * direction
        q2 = p2 - (p2 @ direction) * direction

        d10 = q1 - q0
        d20 = q2 - q0
        d21 = q2 - q1

        # Direction is with plane (p0,p1,p2)
        s = np.fabs(numba_cross(d10, d20) @ direction)

        if numba_close(s, 0.0):
            return np.zeros(3, dtype=np.float64), -1.0, np.zeros(3, dtype=np.float64)

        radius = np.sqrt((d10 @ d10) * (d20 @ d20) * (d21 @ d21)) / (2 * s)

        m = np.vstack((d10, d20, direction))

        n0 = q0 @ q0
        n1 = q1 @ q1
        n2 = q2 @ q2

        b = np.array([0.5 * (n1 - n0), 0.5 * (n2 - n0), 0])
        center = np.linalg.inv(m) @ b

    except Exception:
        return np.zeros(3, dtype=np.float64), -1.0, np.zeros(3, dtype=np.float64)

    return center, radius, direction


@njit(cache=True)
def numba_mark_selected_inliers(
    p: np.ndarray,
    threshold: float,
    center: np.ndarray,
    radius: float,
    direction: np.ndarray,
) -> np.ndarray:
    """
    Compute inliers from a point set p, that lies with threshold distance to the cylinder

    Args:
        p (np.array(dtype=np.float64)): Input point set Nx3
        threshold (float): Max distance to the cylinder (absolute value)
        center (np.array(dtype=np.float64)): The center of the cylinder
        radius (float): The radius of the cylinder
        direction (np.array(dtype=np.float64)): the direction in which the cylinder points

    Returns:
        (np.array(dtype=bool)): Inlier points selected map
    """

    d = np.fabs(numba_distance(p, center, radius, direction))
    return d < threshold


@njit(cache=True)
def numba_filter_points(
    p: np.ndarray, center: np.ndarray, r_min: float, r_max: float
) -> np.ndarray:
    """
    Filters points in p to only keep those that at least at r_min distance from center and at most r_max distance from
    center (r_min and r_max excluded)

    Args:
        p (np.array(dtype=np.float64)): Points to filter, as a Nx3 array
        center (np.array(dtype=np.float64)): Reference point to compute distances from
        r_min (float): Only points with ]r_min,r_max[ distance to center are kept
        r_max (float): Only points with ]r_min,r_max[ distance to center are kept

    Returns:
        np.array(dtype=np.float64): Points that are kept
    """

    d = np.sqrt(np.sum((p - center) ** 2, axis=1))
    i = np.where(np.logical_and(d > r_min, d < r_max))[0]
    return p[i]


@njit(nogil=True, parallel=True, cache=True)
def numba_fit_cylinder_ransac(
    p: np.ndarray,
    axis: np.ndarray,
    nb_test_min: int,
    nb_test_max: int,
    sufficient_pct_inl: float,
    r_min: float,
    r_max: float,
    err: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Fits a cylinder to a set of points using RANSAC, given the direction for the cylinder's axis
    The percentage of inliers might be below pct_inl if nb_test_max is reached.
    In that case, the cylinder with the best percentage of inliers is returned.

    Args:
        p (np.array(dtype=np.float64)): Input set of points
        axis (np.array(dtype=np.float64)): Cylinder's axis
        nb_test_min (int): Min number of RANSAC tests
        nb_test_max (int): Max number of RANSAC tests
        pct_inl (float): Minimum allowable percentage of inliers
        r_min (float): Min radius allowed for returned cylinder
        r_max (float): Max radius allowed for returned cylinder
        err (float): Maximum allowable distance to cylinder for an inlier point

    Returns:
        basis, inliers, pct (tuple[np.ndarray, np.ndarray, float]): Respectively
        the 3 points that define the best cylinder, its inlier points and the percentage of inliers that it represents
    """
    # np.random.seed(42) # important comment, do not delete (used in jit_compiled_functions_test)
    best_basis = np.empty(shape=(0, 0), dtype=np.float64)
    best_inliers = np.empty(shape=(0, 0), dtype=np.float64)
    best_pct = 0.0

    if p.shape[0] < 3:
        return best_basis, best_inliers, best_pct

    thread_cylinder_basis = np.zeros(shape=(nb_test_min, 3), dtype=np.uint64)
    thread_pct_inliers = np.zeros(shape=(nb_test_min), dtype=np.float64)
    # Do this at least nb_test_min times in parallel
    for test_idx in prange(nb_test_min):
        # Randomly pick 3 points
        p_indexes = numba_random_indices(p.shape[0])
        q = p[p_indexes]

        # Fit cylinder
        center, radius, direction = numba_fit_3_points_cylinder(q[0], q[1], q[2], axis)

        if r_min < radius < r_max:
            # Compute inliers
            inliers = numba_mark_selected_inliers(p, err, center, radius, direction)

            # Update each thread value
            thread_cylinder_basis[test_idx] = p_indexes
            thread_pct_inliers[test_idx] = inliers.sum() / inliers.shape[0]

    if thread_pct_inliers.size != 0:
        max_pct_inlier_idx = thread_pct_inliers.argmax()
        best_basis = p[thread_cylinder_basis[max_pct_inlier_idx]]
        best_center, best_radius, best_direction = numba_fit_3_points_cylinder(
            best_basis[0], best_basis[1], best_basis[2], axis
        )
        best_inliers = p[
            numba_mark_selected_inliers(
                p, err, best_center, best_radius, best_direction
            )
        ]
        best_pct = thread_pct_inliers[max_pct_inlier_idx]
        if best_pct >= sufficient_pct_inl:
            # Return if we achieved a suitable result during the nb_test_min's test
            return best_basis, best_inliers, best_pct

    thread_cylinder_basis = np.zeros(shape=(nb_test_max, 3), dtype=np.uint64)
    thread_pct_inliers = np.zeros(shape=(nb_test_max), dtype=np.float64)

    # Flag to tell all the thread that they have to stop searching
    stop_flag = np.array([0], dtype=np.int64)

    nb_batches = 10.0
    execution_per_batch = int(np.ceil(nb_test_max / nb_batches))
    for _ in range(nb_batches):
        # Now go up to nb_test_max tries, but return as soon as a correct cylinder has been found
        # (ie percentage of inliers is sufficient)
        if stop_flag[0] != 0:
            break
        for test_idx in prange(execution_per_batch):
            # Randomly pick 3 points
            p_indexes = numba_random_indices(p.shape[0])
            q = p[p_indexes]

            # Fit cylinder
            center, radius, direction = numba_fit_3_points_cylinder(
                q[0], q[1], q[2], axis
            )

            if r_min < radius < r_max:
                # Compute inliers
                inliers = numba_mark_selected_inliers(p, err, center, radius, direction)

                # Update each thread value
                thread_cylinder_basis[test_idx] = p_indexes
                current_pct_inliers = inliers.sum() / inliers.shape[0]
                thread_pct_inliers[test_idx] = current_pct_inliers

                if current_pct_inliers >= sufficient_pct_inl:
                    atomic_exchange(stop_flag, 1)

    if thread_pct_inliers.size != 0:
        max_pct_inlier_idx = thread_pct_inliers.argmax()
        if thread_pct_inliers[max_pct_inlier_idx] > best_pct:
            best_basis = p[thread_cylinder_basis[max_pct_inlier_idx]]
            best_center, best_radius, best_direction = numba_fit_3_points_cylinder(
                best_basis[0], best_basis[1], best_basis[2], axis
            )
            best_inliers = p[
                numba_mark_selected_inliers(
                    p, err, best_center, best_radius, best_direction
                )
            ]
            best_pct = thread_pct_inliers[max_pct_inlier_idx]
    return best_basis, best_inliers, best_pct


@njit(cache=True)
def numba_close(a: float, b: float) -> bool:
    """
    Numba implementation of math.isclose (numba does not support isclose())

    Return True if the values a and b are close to each other and False otherwise.
    Whether or not two values are considered close is determined according to given absolute and relative tolerances.

    Args:
        a (float): First number
        b (float): Seconc number
    Returns:
        (bool) True if the number are close according to absolute and relative tolerance criterions, False otherwise
    """
    rel_tol = 1e-09
    abs_tol = 0.0
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


@njit(cache=True)
def numba_distance(
    p: np.ndarray, center: np.ndarray, radius: float, direction: np.ndarray
) -> np.ndarray:
    """
    Signed distance to a set of points p. Positive outside the cylinder.

    Args:
        p (np.array(dtype=np.float64)): Nx3 array containing set of points
        center (np.array(dtype=np.float64)): The center of the cylinder
        radius (float): The radius of the cylinder
        direction (np.array(dtype=np.float64)): the direction in which the cylinder points

    Returns:
        float: Signed distance between a cylinder and a set of points
    """

    d = p - center
    li = d @ direction
    dist_to_axis = np.sum(d * d, axis=1) - li * li

    # Handle case of numerical errors causing dist_to_axis to be negative
    dist_to_axis[dist_to_axis < 0] = 0
    dist = np.sqrt(dist_to_axis)

    return dist - radius
