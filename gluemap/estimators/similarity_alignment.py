"""
SIM(3) point-cloud alignment for GlueMap mini-stars.

Adapted from VGGT-Long/loop_utils/sim3utils.py.

Each mini-star independently estimates a dense world-frame point map
(N_imgs, H, W, 3) for every image it contains. When two stars share one or
more images, those shared images provide pixel-to-pixel correspondences
between the two local coordinate frames — no feature matching required.

Typical usage:
    worldpoints_dict = load_worldpoints_dict(pointmap_dir)
    image_to_stars = build_image_to_stars(worldpoints_dict)
    sim3_dict = align_all_edges(worldpoints_dict, predictions_dict)
"""

import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyceres
import pygluemap


# ---------------------------------------------------------------------------
# Section 1 — Core SIM(3) math (ported from VGGT-Long sim3utils.py)
# ---------------------------------------------------------------------------

def estimate_sim3(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Umeyama SIM(3) estimation: find (s, R, t) such that target ≈ s*R@source + t.

    Args:
        source_points: (N, 3)
        target_points: (N, 3)

    Returns:
        s: scale factor (scalar)
        R: (3, 3) rotation matrix
        t: (3,) translation vector
    """
    mu_src = np.mean(source_points, axis=0)
    mu_tgt = np.mean(target_points, axis=0)

    src_centered = source_points - mu_src
    tgt_centered = target_points - mu_tgt

    scale_src = np.sqrt((src_centered ** 2).sum(axis=1).mean())
    scale_tgt = np.sqrt((tgt_centered ** 2).sum(axis=1).mean())
    s = scale_tgt / (scale_src + 1e-12)

    src_scaled = src_centered * s
    H = src_scaled.T @ tgt_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = mu_tgt - s * R @ mu_src
    return s, R, t


def weighted_estimate_sim3(
    source_points: np.ndarray,
    target_points: np.ndarray,
    weights: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Weighted SIM(3) estimation.

    Args:
        source_points: (N, 3)
        target_points: (N, 3)
        weights:       (N,) non-negative weights

    Returns:
        s, R, t
    """
    total_weight = np.sum(weights)
    if total_weight < 1e-6:
        raise ValueError("Total weight too small for meaningful estimation")

    w = weights / total_weight

    mu_src = np.sum(w[:, None] * source_points, axis=0)
    mu_tgt = np.sum(w[:, None] * target_points, axis=0)

    src_centered = source_points - mu_src
    tgt_centered = target_points - mu_tgt

    scale_src = np.sqrt(np.sum(w * np.sum(src_centered ** 2, axis=1)))
    scale_tgt = np.sqrt(np.sum(w * np.sum(tgt_centered ** 2, axis=1)))
    s = scale_tgt / (scale_src + 1e-12)

    H = (s * src_centered * np.sqrt(w)[:, None]).T @ (tgt_centered * np.sqrt(w)[:, None])
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = mu_tgt - s * R @ mu_src
    return s, R, t


def robust_weighted_estimate_sim3(
    src: np.ndarray,
    tgt: np.ndarray,
    init_weights: np.ndarray,
    delta: float = 0.1,
    max_iters: int = 20,
    tol: float = 1e-9,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Robust SIM(3) via Iteratively Reweighted Least Squares (IRLS) with Huber loss.

    Args:
        src:          (N, 3) source points
        tgt:          (N, 3) target points
        init_weights: (N,) initial weights
        delta:        Huber loss threshold
        max_iters:    maximum IRLS iterations
        tol:          convergence tolerance

    Returns:
        s, R, t
    """
    s, R, t = weighted_estimate_sim3(src, tgt, init_weights)
    prev_error = float("inf")

    for _ in range(max_iters):
        transformed = s * (src @ R.T) + t
        residuals = np.linalg.norm(tgt - transformed, axis=1)

        huber_weights = np.ones_like(residuals)
        large = residuals > delta
        huber_weights[large] = delta / (residuals[large] + 1e-12)

        combined = init_weights * huber_weights
        combined /= np.sum(combined) + 1e-12

        s_new, R_new, t_new = weighted_estimate_sim3(src, tgt, combined)

        param_change = abs(s_new - s) + np.linalg.norm(t_new - t)
        rot_angle = np.arccos(
            min(1.0, max(-1.0, (np.trace(R_new @ R.T) - 1) / 2))
        )
        abs_res = np.abs(residuals)
        huber_loss = np.where(abs_res <= delta, 0.5 * abs_res ** 2, delta * (abs_res - 0.5 * delta))
        current_error = np.sum(huber_loss * init_weights)

        if (param_change < tol and rot_angle < np.radians(0.1)) or (
            abs(prev_error - current_error) < tol * (prev_error + 1e-12)
        ):
            s, R, t = s_new, R_new, t_new
            break

        s, R, t = s_new, R_new, t_new
        prev_error = current_error

    return s, R, t


def apply_sim3(
    points: np.ndarray,
    s: float,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Apply SIM(3) to a set of points. (N, 3) -> (N, 3)."""
    return (s * (R @ points.T)).T + t


def apply_sim3_direct(
    point_maps: np.ndarray,
    s: float,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """
    Apply SIM(3) to a batch of dense point maps.

    Args:
        point_maps: (b, H, W, 3)  or  (H, W, 3)

    Returns:
        transformed point maps, same shape as input
    """
    expanded = point_maps[..., np.newaxis]          # (..., 3, 1)
    rotated = np.matmul(R, expanded).squeeze(-1)    # (..., 3)
    return s * rotated + t


def accumulate_sim3_transforms(
    transforms: List[Tuple[float, np.ndarray, np.ndarray]],
) -> List[Tuple[float, np.ndarray, np.ndarray]]:
    """
    Convert a list of adjacent SIM(3) transforms to cumulative transforms
    from the first frame.

    Args:
        transforms: [(s0,R0,t0), (s1,R1,t1), ...] — relative transforms

    Returns:
        cumulative list where element k is the transform from frame 0 to frame k
    """
    if not transforms:
        return []

    cumulative = [transforms[0]]
    for s_next, R_next, t_next in transforms[1:]:
        s_prev, R_prev, t_prev = cumulative[-1]
        s_cum = s_prev * s_next
        R_cum = R_prev @ R_next
        t_cum = s_prev * (R_prev @ t_next) + t_prev
        cumulative.append((s_cum, R_cum, t_cum))

    return cumulative


def compute_alignment_error(
    point_map1: np.ndarray,
    point_map2: np.ndarray,
    s: float,
    R: np.ndarray,
    t: np.ndarray,
) -> float:
    """
    Compute mean alignment error after applying (s, R, t) to point_map2.

    Args:
        point_map1: (N, 3) or (b, H, W, 3) target
        point_map2: (N, 3) or (b, H, W, 3) source
        s, R, t: SIM(3) transform

    Returns:
        mean_error (float), also prints statistics
    """
    pts1 = point_map1.reshape(-1, 3)
    pts2 = point_map2.reshape(-1, 3)

    # Filter NaN/inf
    valid = np.isfinite(pts1).all(axis=1) & np.isfinite(pts2).all(axis=1)
    pts1, pts2 = pts1[valid], pts2[valid]

    if len(pts1) == 0:
        print("Warning: no valid points for error computation")
        return float("nan")

    transformed = apply_sim3(pts2, s, R, t)
    errors = np.linalg.norm(transformed - pts1, axis=1)

    mean_e = np.mean(errors)
    print(
        f"Alignment error [{len(errors)} pts]: "
        f"mean={mean_e:.4f}, std={np.std(errors):.4f}, "
        f"median={np.median(errors):.4f}, max={np.max(errors):.4f}"
    )
    return mean_e


# ---------------------------------------------------------------------------
# Section 2 — World-points loading
# ---------------------------------------------------------------------------

def load_worldpoints_dict(pointmap_dir: str) -> Dict[int, Dict]:
    """
    Load all ``star_{N:05d}.npz`` files from *pointmap_dir*.

    Returns:
        worldpoints_dict[idx_star] = {
            "world_points": np.ndarray(N_imgs, H, W, 3),
            "indexes":      list[int],  # image IDs, len = N_imgs
        }
    """
    worldpoints_dict: Dict[int, Dict] = {}

    for path in sorted(glob.glob(os.path.join(pointmap_dir, "star_*.npz"))):
        basename = os.path.splitext(os.path.basename(path))[0]  # "star_00003"
        idx_star = int(basename.split("_")[1])
        data = np.load(path)
        worldpoints_dict[idx_star] = {
            "world_points": data["world_points"],          # (N, H, W, 3) float32
            "indexes": data["indexes"].tolist(),            # list[int]
        }

    return worldpoints_dict


def build_image_to_stars(
    worldpoints_dict: Dict[int, Dict],
) -> Dict[int, List[Tuple[int, int]]]:
    """
    Build a reverse mapping from image_id to the (star, position) pairs that contain it.

    Returns:
        image_to_stars[image_id] = [(idx_star, img_pos), ...]
    """
    image_to_stars: Dict[int, List[Tuple[int, int]]] = {}
    for idx_star, data in worldpoints_dict.items():
        for img_pos, image_id in enumerate(data["indexes"]):
            image_to_stars.setdefault(image_id, []).append((idx_star, img_pos))
    return image_to_stars


# ---------------------------------------------------------------------------
# Section 3 — Per-edge SIM(3) alignment
# ---------------------------------------------------------------------------

def align_point_maps(
    point_map1: np.ndarray,
    point_map2: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Align two dense world-frame point maps: estimate SIM(3) mapping point_map2 → point_map1.

    Args:
        point_map1: (b, H, W, 3) or (H, W, 3)  target (star A's estimate)
        point_map2: (b, H, W, 3) or (H, W, 3)  source (star B's estimate)

    Returns:
        s, R, t  such that  p1 ≈ s*R@p2 + t
    """
    pts1 = point_map1.reshape(-1, 3)
    pts2 = point_map2.reshape(-1, 3)

    valid = np.isfinite(pts1).all(axis=1) & np.isfinite(pts2).all(axis=1)
    pts1, pts2 = pts1[valid], pts2[valid]

    if len(pts1) == 0:
        raise ValueError("No finite point pairs found for alignment")

    print(f"align_point_maps: {len(pts1)} point correspondences")
    s, R, t = estimate_sim3(pts2, pts1)
    compute_alignment_error(pts1, pts2, s, R, t)
    return s, R, t


def weighted_align_point_maps(
    point_map1: np.ndarray,
    point_map2: np.ndarray,
    delta: float = 0.1,
    max_iters: int = 20,
    tol: float = 1e-9,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Robustly align two dense point maps using IRLS with uniform initial weights.

    Args:
        point_map1: target, (b, H, W, 3) or (H, W, 3)
        point_map2: source, (b, H, W, 3) or (H, W, 3)

    Returns:
        s, R, t
    """
    pts1 = point_map1.reshape(-1, 3)
    pts2 = point_map2.reshape(-1, 3)

    valid = np.isfinite(pts1).all(axis=1) & np.isfinite(pts2).all(axis=1)
    pts1, pts2 = pts1[valid], pts2[valid]

    if len(pts1) == 0:
        raise ValueError("No finite point pairs found for alignment")

    print(f"weighted_align_point_maps: {len(pts1)} point correspondences")
    weights = np.ones(len(pts1), dtype=np.float64)
    s, R, t = robust_weighted_estimate_sim3(pts2, pts1, weights, delta=delta,
                                            max_iters=max_iters, tol=tol)
    compute_alignment_error(pts1, pts2, s, R, t)
    return s, R, t


def align_star_pair(
    worldpoints_dict: Dict[int, Dict],
    idx_star_A: int,
    idx_star_B: int,
    shared_image_ids: Optional[List[int]] = None,
    use_robust: bool = True,
    delta: float = 0.1,
    max_iters: int = 20,
    tol: float = 1e-9,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate SIM(3) from star_A's coordinate frame to star_B's frame using
    the world-point maps of all images shared between the two stars.

    Each shared image gives H*W independent point correspondences — one per pixel.
    This leverages the fact that both stars independently estimated the same scene
    from different centers (the "independently estimated twice" property).

    Args:
        worldpoints_dict: loaded from load_worldpoints_dict()
        idx_star_A:       source star index
        idx_star_B:       target star index
        shared_image_ids: if None, auto-detected from the index lists
        use_robust:       use IRLS (True) or plain Umeyama (False)

    Returns:
        (s, R, t) such that  p_B ≈ s * R @ p_A + t
    """
    data_A = worldpoints_dict[idx_star_A]
    data_B = worldpoints_dict[idx_star_B]

    ids_A = set(data_A["indexes"])
    ids_B = set(data_B["indexes"])

    if shared_image_ids is None:
        shared = list(ids_A & ids_B)
    else:
        shared = [img_id for img_id in shared_image_ids if img_id in ids_A and img_id in ids_B]

    if not shared:
        raise ValueError(
            f"Stars {idx_star_A} and {idx_star_B} share no images; cannot align."
        )

    all_pts_A = []
    all_pts_B = []

    for img_id in shared:
        pos_A = data_A["indexes"].index(img_id)
        pos_B = data_B["indexes"].index(img_id)

        wp_A = data_A["world_points"][pos_A]  # (H, W, 3)
        wp_B = data_B["world_points"][pos_B]  # (H, W, 3)

        pts_A = wp_A.reshape(-1, 3)
        pts_B = wp_B.reshape(-1, 3)

        valid = np.isfinite(pts_A).all(axis=1) & np.isfinite(pts_B).all(axis=1)
        all_pts_A.append(pts_A[valid])
        all_pts_B.append(pts_B[valid])

    all_pts_A = np.concatenate(all_pts_A, axis=0)
    all_pts_B = np.concatenate(all_pts_B, axis=0)

    print(
        f"align_star_pair({idx_star_A}, {idx_star_B}): "
        f"{len(shared)} shared image(s), {len(all_pts_A)} point pairs"
    )

    if use_robust:
        weights = np.ones(len(all_pts_A), dtype=np.float64)
        s, R, t = robust_weighted_estimate_sim3(
            all_pts_A, all_pts_B, weights, delta=delta, max_iters=max_iters, tol=tol
        )
    else:
        s, R, t = estimate_sim3(all_pts_A, all_pts_B)

    compute_alignment_error(all_pts_B, all_pts_A, s, R, t)
    return s, R, t


def align_all_edges(
    worldpoints_dict: Dict[int, Dict],
    predictions_dict: Optional[Dict] = None,
    use_robust: bool = True,
    delta: float = 0.1,
    max_iters: int = 20,
    tol: float = 1e-9,
) -> Dict[Tuple[int, int], Tuple[float, np.ndarray, np.ndarray]]:
    """
    Estimate SIM(3) for every pair of stars that share at least one image.

    Both directions (A→B) and (B→A) are stored. The reverse uses the inverse:
        s_BA = 1/s_AB,  R_BA = R_AB.T,  t_BA = -s_BA * R_BA @ t_AB

    Convention: idx_star_A is source, idx_star_B is target.
    sim3_dict[(A, B)] = (s, R, t) such that p_B ≈ s * R @ p_A + t.

    Args:
        worldpoints_dict: loaded from load_worldpoints_dict()
        predictions_dict: optional, not used for alignment but kept for API parity

    Returns:
        sim3_dict[(idx_star_A, idx_star_B)] = (s, R, t)
            aligning star_A's frame into star_B's frame
    """
    # Build a mapping from center image_id → star index
    center_to_star: Dict[int, int] = {}
    for idx_star, data in worldpoints_dict.items():
        center_id = data["indexes"][0]
        center_to_star[center_id] = idx_star

    # Only pair stars where one star's neighbor is the center of another star
    star_pairs: set = set()
    for idx_star, data in worldpoints_dict.items():
        for neighbor_id in data["indexes"][1:]:
            if neighbor_id in center_to_star:
                other_star = center_to_star[neighbor_id]
                if other_star != idx_star:
                    star_pairs.add((min(idx_star, other_star), max(idx_star, other_star)))

    sim3_dict: Dict[Tuple[int, int], Tuple[float, np.ndarray, np.ndarray]] = {}

    for idx_A, idx_B in sorted(star_pairs):
        try:
            s, R, t = align_star_pair(
                worldpoints_dict, idx_A, idx_B,
                use_robust=use_robust, delta=delta, max_iters=max_iters, tol=tol,
            )
            sim3_dict[(idx_A, idx_B)] = (s, R, t)

            # Inverse: star_A → star_B
            s_inv = 1.0 / (s + 1e-12)
            R_inv = R.T
            t_inv = -s_inv * (R_inv @ t)
            sim3_dict[(idx_B, idx_A)] = (s_inv, R_inv, t_inv)

        except ValueError as e:
            print(f"Skipping pair ({idx_A}, {idx_B}): {e}")

    print(f"align_all_edges: estimated {len(star_pairs)} edge(s), "
          f"{len(sim3_dict)} directional transforms")
    return sim3_dict


# ---------------------------------------------------------------------------
# Section 4 — Motion averaging (rotation + scale + translation)
# ---------------------------------------------------------------------------

def scale_averaging(
    sim3_dict: Dict[Tuple[int, int], Tuple[float, np.ndarray, np.ndarray]],
    num_stars: int,
) -> List[np.ndarray]:
    """
    Estimate globally consistent scales for each mini-star from pairwise SIM(3) constraints.

    Uses a single Ceres problem with DepthConsistencySameCamError(1.0, s_AB) which encodes
    the constraint  log(S_B / S_A) = log(s_AB)  for each directed edge (A < B).

    Args:
        sim3_dict:  output of align_all_edges() — (s, R, t) per directed edge
        num_stars:  total number of mini-stars

    Returns:
        scales: List[np.ndarray(1,)] of length num_stars — same format as
                global_scales in similarity_averaging.py (star 0 anchored to 1.0)
    """

    scales = [np.ones(1, dtype=np.float64) for _ in range(num_stars)]
    prob = pyceres.Problem()

    for (A, B), (s_AB, _R, _t) in sim3_dict.items():
        if A >= B:
            continue  # process each undirected edge once
        if A >= num_stars or B >= num_stars:
            continue
        cost = pygluemap.DepthConsistencySameCamError(1.0, float(s_AB))
        prob.add_residual_block(cost, None, [scales[A], scales[B]])

    for i in range(num_stars):
        if not prob.has_parameter_block(scales[i]):
            continue
        prob.set_parameter_lower_bound(scales[i], 0, 1e-5)
        prob.set_parameter_upper_bound(scales[i], 0, 1e5)

    # Anchor star 0 to scale = 1.0
    if prob.has_parameter_block(scales[0]):
        prob.set_parameter_block_constant(scales[0])

    options = pyceres.SolverOptions()
    options.linear_solver_type = pyceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY
    options.num_threads = 32
    options.max_num_iterations = 100
    options.minimizer_progress_to_stdout = False

    summary = pyceres.SolverSummary()
    pyceres.solve(options, prob, summary)
    print("Scale averaging:", summary.BriefReport())

    return scales


def rotation_averaging_from_sim3(
    sim3_dict: Dict[Tuple[int, int], Tuple[float, np.ndarray, np.ndarray]],
    num_stars: int,
) -> Dict[int, np.ndarray]:
    """
    Rotation averaging over star-level pairwise rotations from sim3_dict.

    Mirrors rotation_averaging() in gluemap/estimators/rotation_averaging.py but
    nodes are star indices (0..num_stars-1) and edges come from sim3_dict.

    Args:
        sim3_dict:  pairwise SIM(3) from align_all_edges()
        num_stars:  total number of mini-stars

    Returns:
        Dict[star_idx, np.ndarray(3,3)] — globally consistent rotation per star
    """
    import pyceres
    import pygluemap
    from gluemap.math.geometry import rotation_matrix_to_quaternion, quaternion_to_rotation_matrix

    rotations = {i: np.array([1., 0., 0., 0.], dtype=np.float64) for i in range(num_stars)}
    prob = pyceres.Problem()

    for (A, B), (s_AB, R_AB, t_AB) in sim3_dict.items():
        if A >= B:
            continue  # undirected: add each edge once
        q_AB = rotation_matrix_to_quaternion(R_AB)
        cost = pygluemap.RotationGeodesicError(q_AB)
        prob.add_residual_block(cost, None, [rotations[A], rotations[B]])

    for i, q in rotations.items():
        if prob.has_parameter_block(q):
            prob.set_manifold(q, pyceres.QuaternionManifold())

    if prob.has_parameter_block(rotations[0]):
        prob.set_parameter_block_constant(rotations[0])

    options = pyceres.SolverOptions()
    options.linear_solver_type = (
        pyceres.LinearSolverType.DENSE_QR
        if num_stars < 200
        else pyceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY
    )
    options.num_threads = -1
    options.max_num_iterations = 200
    options.minimizer_progress_to_stdout = False

    summary = pyceres.SolverSummary()
    pyceres.solve(options, prob, summary)
    print("Rotation averaging (sim3):", summary.BriefReport())

    for i in rotations:
        rotations[i] = quaternion_to_rotation_matrix(rotations[i])
    return rotations


def similarity_averaging_from_sim3(
    sim3_dict: Dict[Tuple[int, int], Tuple[float, np.ndarray, np.ndarray]],
    global_rotations: Dict[int, np.ndarray],
    global_centers: Optional[Dict[int, np.ndarray]] = None,
    global_scales: Optional[List[np.ndarray]] = None,
    num_stars: Optional[int] = None,
    max_num_iterations: int = 50,
    fix_scales: bool = False,
) -> Dict[int, np.ndarray]:
    """
    Similarity averaging over star-level pairwise SIM(3) from sim3_dict.

    Mirrors similarity_averaging() / add_star_edge_error() in
    gluemap/estimators/similarity_averaging.py but uses sim3_dict edges between
    stars instead of predictions_dict edges between images.

    Translation observation by analogy with add_star_edge_error's formula
    s_i*(c_j - c_i) = -R_j^T @ t_ij:
        t_obs = -global_rotations[A].T @ t_AB
    where A is the source and B is the target (convention: p_B = s*R@p_A + t).

    Args:
        sim3_dict:         pairwise SIM(3) from align_all_edges()
        global_rotations:  Dict[star_idx, R(3,3)] from rotation_averaging_from_sim3()
        global_centers:    initial centers (zeros if None)
        global_scales:     initial scales (ones if None)
        num_stars:         total number of stars (inferred if None)
        max_num_iterations: Ceres solver iterations
        fix_scales:        if True, pin all scale blocks constant

    Returns:
        Dict[star_idx, np.ndarray(3,)] — globally consistent center per star
    """
    import pyceres
    import pygluemap

    if num_stars is None:
        num_stars = max(max(A, B) for A, B in sim3_dict) + 1

    if global_centers is None:
        global_centers = {i: np.zeros(3, dtype=np.float64) for i in range(num_stars)}
    if global_scales is None:
        global_scales = [np.ones(1, dtype=np.float64) for _ in range(num_stars)]

    prob = pyceres.Problem()
    first_anchor = True

    for (A, B), (s_AB, R_AB, t_AB) in sim3_dict.items():
        if A >= B:
            continue
        t_obs = (-global_rotations[B].T @ t_AB.reshape(3, 1)).flatten()
        cost = pygluemap.BATAPairwiseDirectionError(t_obs)
        prob.add_residual_block(
            cost, None, [global_centers[A], global_centers[B], global_scales[A]]
        )
        if first_anchor:
            prob.set_parameter_block_constant(global_centers[A])
            prob.set_parameter_block_constant(global_scales[A])
            first_anchor = False

    for i in range(num_stars):
        if not prob.has_parameter_block(global_scales[i]):
            continue
        if fix_scales:
            prob.set_parameter_block_constant(global_scales[i])
        else:
            prob.set_parameter_lower_bound(global_scales[i], 0, 1e-5)
            prob.set_parameter_upper_bound(global_scales[i], 0, 1e5)

    options = pyceres.SolverOptions()
    options.linear_solver_type = pyceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY
    options.num_threads = 32
    options.max_num_iterations = max_num_iterations
    options.minimizer_progress_to_stdout = False

    summary = pyceres.SolverSummary()
    pyceres.solve(options, prob, summary)
    print("Similarity averaging (sim3):", summary.BriefReport())

    return global_centers


def motion_averaging(
    sim3_dict: Dict[Tuple[int, int], Tuple[float, np.ndarray, np.ndarray]],
    predictions_dict: Dict,
    num_stars: Optional[int] = None,
) -> Tuple[Dict, List[np.ndarray], Dict]:
    """
    Recover globally consistent absolute poses for all mini-stars in three steps:

      1. Rotation averaging  — rotation_averaging_from_sim3()
      2. Scale averaging     — scale_averaging()
      3. Translation averaging — similarity_averaging_from_sim3(..., fix_scales=True)

    Results are re-keyed from star indices to center image IDs using
    predictions_dict["indexes"][k][0], since each image is guaranteed to be
    the center of exactly one star.

    Args:
        sim3_dict:        pairwise SIM(3) from align_all_edges()
        predictions_dict: star inference output (indexes, extrinsics, pose_scores, …)
        num_stars:        total number of stars (inferred if None)

    Returns:
        global_rotations:   Dict[image_id, np.ndarray(3,3)]
        global_scales:      List[np.ndarray(1,)]  — one entry per star
        global_centers:     Dict[image_id, np.ndarray(3,)]
    """
    if num_stars is None:
        num_stars = len(predictions_dict["indexes"])

    print("=== motion_averaging: Step 1 — rotation averaging ===")
    global_rotations_star = rotation_averaging_from_sim3(sim3_dict, num_stars)

    print("=== motion_averaging: Step 2 — scale averaging ===")
    global_scales = scale_averaging(sim3_dict, num_stars)

    print("=== motion_averaging: Step 3 — translation averaging (fixed scales) ===")
    global_centers_star = similarity_averaging_from_sim3(
        sim3_dict,
        global_rotations_star,
        global_scales=global_scales,
        num_stars=num_stars,
        max_num_iterations=50,
        fix_scales=True,
    )

    # Re-key: star_idx → center image_id
    # Each image is the center of exactly one star.
    global_rotations = {
        predictions_dict["indexes"][k][0]: global_rotations_star[k]
        for k in range(num_stars)
    }
    global_centers = {
        predictions_dict["indexes"][k][0]: global_centers_star[k]
        for k in range(num_stars)
    }

    return global_rotations, global_scales, global_centers
