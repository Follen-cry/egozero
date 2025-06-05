"""Utilities for extracting depth values of points"""

import numpy as np
from scipy.optimize import least_squares


def triangulate_linear_multi_view(uvs, Rt_list, K):
    """Triangulate from multiple views using linear DLT"""
    A = []
    for uv, Rt in zip(uvs, Rt_list):
        P = K @ Rt[:3]
        A.append(uv[0] * P[2] - P[0])
        A.append(uv[1] * P[2] - P[1])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]


def project(K, Rt, X):
    """Project 3D point X into image using camera intrinsics/extrinsics"""
    X_cam = Rt @ np.append(X, 1.0)
    x = X_cam[:3] / X_cam[2]
    uv = K @ x
    return uv[:2]


def filter_views_by_epipolar_consistency(
    tracked_uvs, Rt_list, K, threshold=2.0, min_consistent_views=5
):
    """Filter views that are not consistent with others under epipolar geometry."""
    num_views = len(tracked_uvs)
    consistent_counts = np.zeros(num_views, dtype=int)
    inlier_mask = np.ones(num_views, dtype=bool)

    for i in range(num_views):
        for j in range(num_views):
            if i == j:
                continue
            uv_i = tracked_uvs[i]
            uv_j = tracked_uvs[j]
            Rt_i = Rt_list[i]
            Rt_j = Rt_list[j]

            # Compute fundamental matrix from i to j
            E = Rt_j[:3, :3] @ Rt_i[:3, :3].T
            t = Rt_j[:3, 3] - E @ Rt_i[:3, 3]
            tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
            F = np.linalg.inv(K).T @ tx @ E @ np.linalg.inv(K)

            # Epipolar constraint: uv_j.T * F * uv_i ≈ 0
            uv_i_h = np.array([*uv_i, 1.0])
            uv_j_h = np.array([*uv_j, 1.0])
            epi_error = np.abs(uv_j_h @ F @ uv_i_h)

            if epi_error < threshold:
                consistent_counts[i] += 1

    for i in range(num_views):
        if consistent_counts[i] < min_consistent_views:
            inlier_mask[i] = False

    return inlier_mask


def robust_ransac_triangulation(
    tracked_uvs,
    k,
    c2ws,
    ransac_thresh=5.0,
    ransac_iters=1000,
    views_per_sample=3,
    lower_bound=[-2, -2, 1e-4],
    upper_bound=[2, 2, 10],
    epipolar_thresh=3.0,  # Relaxed epipolar threshold
    min_consistent_pairs=5,
):
    lower_bound = np.array(lower_bound)
    upper_bound = np.array(upper_bound)

    c2ws = np.einsum("ij,njk->nik", np.linalg.inv(c2ws[0]), c2ws)
    Rt_w2c = np.linalg.inv(c2ws)

    # Epipolar filtering of individual views
    mask = filter_views_by_epipolar_consistency(
        tracked_uvs,
        Rt_w2c,
        k,
        threshold=epipolar_thresh,
        min_consistent_views=min_consistent_pairs,
    )

    best_inliers = []
    best_X = None

    for _ in range(ransac_iters):
        idxs = np.random.choice(len(tracked_uvs), size=views_per_sample, replace=False)
        uvs = [tracked_uvs[i] for i in idxs]
        Rts = [Rt_w2c[i] for i in idxs]

        try:
            X_candidate = triangulate_linear_multi_view(uvs, Rts, k)
        except np.linalg.LinAlgError:
            continue

        if not np.all((lower_bound <= X_candidate) & (X_candidate <= upper_bound)):
            continue

        inliers = []
        for i in range(len(tracked_uvs)):
            if not mask[i]:
                continue

            uv_proj = project(k, Rt_w2c[i], X_candidate)
            error = np.linalg.norm(tracked_uvs[i] - uv_proj)
            if error < ransac_thresh:
                inliers.append(i)

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_X = X_candidate

    if best_X is None or len(best_inliers) < views_per_sample:
        raise RuntimeError("RANSAC failed to find a good solution.")

    def residual_fn(X):
        residuals = []
        for i in best_inliers:
            uv_proj = project(k, Rt_w2c[i], X)
            error = tracked_uvs[i] - uv_proj
            delta = 2.0
            norm = np.linalg.norm(error)
            if norm <= delta:
                residuals.extend(error)
            else:
                residuals.extend(delta * error / norm)

        z_weight = 0.1
        residuals.append(z_weight * X[2])
        return np.array(residuals)

    result = least_squares(
        residual_fn,
        np.clip(best_X, lower_bound, upper_bound),
        method="trf",
        bounds=(lower_bound, upper_bound),  # bounds for the solved t* solution
        loss="soft_l1",
    )

    # compute reprojection loss and return the projected points
    uv_projs = []
    xyz_projs = []
    reprojection_cost = 0
    reprojection_cost_inliers = 0
    for i in range(len(c2ws)):
        c2w = c2ws[i]
        w2c = np.linalg.inv(c2w)
        X_c = w2c[:3, :3] @ result.x + w2c[:3, 3]
        xyz_projs.append(X_c)

        uv_proj = k @ (X_c / X_c[2])
        uv_proj = uv_proj[:2]
        uv_projs.append(uv_proj)

        uv_obs = tracked_uvs[i]
        l2 = np.linalg.norm(uv_proj - uv_obs).item()
        reprojection_cost += l2 / len(c2ws)

        if i in best_inliers:
            reprojection_cost_inliers += l2 / len(best_inliers)

    return {
        "t*": np.array(result.x),
        "num_points": len(c2ws),
        "reprojection_cost": reprojection_cost,
        "num_points_inliers": len(best_inliers),
        "reprojection_cost_inliers": reprojection_cost_inliers,
        "reprojected_uvs": np.array(uv_projs),
        "reprojected_xyzs": np.array(xyz_projs),
    }
