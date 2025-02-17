from numba.core.typing.templates import signature
from numba.extending import intrinsic
import numpy as np
from numba import njit, prange, types

"""
Make sure to put all the JIT compiled functions in the same file, cache invalidation will fail in the case of
nested JIT compiled functions in different files. (C.F https://numba.readthedocs.io/en/stable/user/jit.html --> caching)
"""


@njit(cache=True)
def numba_random_indices(n):
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
def atomic_xchg(typingctx, ptr, val):
    sig = signature(types.int64, types.CPointer(types.int64), types.int64)

    def codegen(context, builder, signature, args):
        ptr, val = args

        old = builder.atomic_rmw("xchg", ptr, val, "monotonic")
        return old

    return sig, codegen


@njit
def atomic_exchange(arr, new_value):
    # Convert the array into an array of type CPointer
    ptr = arr.ctypes
    old = atomic_xchg(ptr, new_value)
    return old


@njit(cache=True)
def numba_cross(a, b):
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
def numba_fit_3_points_cylinder(p0, p1, p2, direction):
    """
    Determine the cylinder going through 3 points p0,p1 and p2 and whose axis is along direction.
    Note that if direction were not given, 5 points would be required.

    Args:
        p0 (np.array(dtype=np.float64)): First point to use
        p1 (np.array(dtype=np.float64)): Second point to use
        p2 (np.array(dtype=np.float64)): Third point to use
        direction (np.array(dtype=np.float64)): Direction to have cylinder axis

    Returns:

    """

    try:
        # Normalize direction (if possible)
        n = np.linalg.norm(direction)
        if numba_close(n, 0):
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

        if numba_close(s, 0):
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
def numba_mark_selected_inliers(p, threshold, center, radius, direction):
    """
    Compute inliers from a point set p, that lies with threshold distance to the cylinder

    Args:
        p (np.array(dtype=np.float64)): Input point set Nx3
        threshold (float): Max distance to the cylinder (absolute value)

    Returns:
        (np.array(dtype=bool)): Inlier points selected map
    """

    d = np.fabs(numba_distance(p, center, radius, direction))
    return d < threshold


@njit(nogil=True, parallel=True, cache=True)
def numba_fit_cylinder_ransac(
    p, axis, nb_test_min, nb_test_max, sufficient_pct_inl, r_min, r_max, err
):
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
        cylinder: Fitted cylinder
        np.array(dtype=np.float64): Fitted cylinder's inlier set
    """

    best_basis = np.empty(shape=(0, 0), dtype=np.float64)
    best_inliers = np.empty(shape=(0, 0), dtype=np.float64)
    best_pct = 0.0

    if p.shape[0] < 3:
        return best_basis, best_inliers, best_pct

    thread_cylinder_basis = np.zeros(shape=(nb_test_min, 3), dtype=np.uint64)
    thread_pct_inliers = np.zeros(shape=(nb_test_min), dtype=np.float64)
    # Do this at least nb_test_min times
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
def numba_close(a, b):
    rel_tol = 1e-09
    abs_tol = 0.0
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


@njit(cache=True)
def numba_distance(p, center, radius, direction):
    """
    Signed distance to a set of points p. Positive outside the cylinder.

    Args:
        p (np.array(dtype=np.float64)): Nx3 array containing set of points

    Returns:
        int: Distance between cylinder's center and the set of points
    """

    d = p - center
    li = d @ direction
    dist_to_axis = np.sum(d * d, axis=1) - li * li

    # Handle case of numerical errors causing dist_to_axis to be negative
    dist_to_axis[dist_to_axis < 0] = 0
    dist = np.sqrt(dist_to_axis)

    return dist - radius
