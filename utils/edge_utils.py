import numpy as np
from scipy.ndimage import gaussian_filter1d

EDGE_MODES = {"upper_left", "lower_right"}


def sample_line_evenly(pts, n=100):
    """Resample an open polyline to exactly n evenly-spaced points along its length.

    Args:
        pts (list): Line points as [[x, y], ...].
        n (int, optional): Number of output points. Defaults to 100.

    Returns:
        np.ndarray: Resampled points of shape (n, 2), dtype float64.
    """
    pts = np.array(pts, dtype=np.float64)
    if len(pts) < 2:
        return np.zeros((n, 2), dtype=np.float64)
    dists  = np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
    dists  = np.insert(dists, 0, 0)
    total  = dists[-1]
    if total == 0:
        return np.repeat(pts[:1], n, axis=0)
    sample_d = np.linspace(0, total, n, endpoint=True)
    rx = np.interp(sample_d, dists, pts[:, 0])
    ry = np.interp(sample_d, dists, pts[:, 1])
    return np.column_stack([rx, ry])


def optimize_line_normals(prob_field, reference_pts, target_img_u8,
                          edge_mode, search_range=12, active_indices=None,
                          prob_threshold=0.5, distance_weight=0.3):
    """Refine open-stroke points by searching along outward normals for the highest-scoring position.

    Args:
        prob_field (np.ndarray): Per-pixel probability field of shape (h, w).
        reference_pts (list): Line points as [[x, y], ...].
        target_img_u8 (np.ndarray): Grayscale target frame, dtype uint8.
        edge_mode (str): 'upper_left' or 'lower_right', controls normal direction.
        search_range (int, optional): Normal-search radius in pixels. Defaults to 12.
        active_indices (array-like, optional): Indices of points to refine; others pass through unchanged. Defaults to None (all points).
        prob_threshold (float, optional): Probability above which a confident match is preferred over the raw best score. Defaults to 0.5.
        distance_weight (float, optional): Penalty weight for distance from the original point when selecting a confident match. Defaults to 0.3.

    Returns:
        list[list[int]]: Refined points as [[x, y], ...].
    """
    h, w    = prob_field.shape
    pts_arr = np.array(reference_pts, dtype=np.float32)
    n       = len(pts_arr)

    active_set = set(active_indices) if active_indices is not None else set(range(n))

    pad  = search_range + 2
    xmin_c = max(0, int(pts_arr[:, 0].min()) - pad)
    xmax_c = min(w, int(pts_arr[:, 0].max()) + pad + 1)
    ymin_c = max(0, int(pts_arr[:, 1].min()) - pad)
    ymax_c = min(h, int(pts_arr[:, 1].max()) + pad + 1)

    pf_crop  = prob_field[ymin_c:ymax_c, xmin_c:xmax_c]
    img_crop = target_img_u8[ymin_c:ymax_c, xmin_c:xmax_c]
    ch, cw   = pf_crop.shape

    grad_y, grad_x = np.gradient(pf_crop)

    pts_loc = pts_arr - np.array([xmin_c, ymin_c], dtype=np.float32)

    prev_idx = np.maximum(np.arange(n) - 1, 0)
    next_idx = np.minimum(np.arange(n) + 1, n - 1)

    tang     = pts_loc[next_idx] - pts_loc[prev_idx]
    tang_len = np.hypot(tang[:, 0], tang[:, 1]) + 1e-8
    tang    /= tang_len[:, None]

    normals  = np.stack([tang[:, 1], -tang[:, 0]], axis=1)
    if edge_mode == 'upper_left':
        flip = normals[:, 1] < 0
    else:
        flip = normals[:, 1] > 0
    normals[flip] *= -1.0

    active_arr = np.array(sorted(active_set), dtype=np.int64)
    inactive   = np.ones(n, dtype=bool)
    inactive[active_arr] = False

    steps = np.arange(-search_range, search_range + 1, dtype=np.float32)
    nA    = len(active_arr)

    pts_a = pts_loc[active_arr]
    nrm_a = normals[active_arr]

    sx_all = pts_a[:, 0:1] + nrm_a[:, 0:1] * steps[None, :]
    sy_all = pts_a[:, 1:2] + nrm_a[:, 1:2] * steps[None, :]

    ix_all = np.round(sx_all).astype(np.int32)
    iy_all = np.round(sy_all).astype(np.int32)

    valid  = (ix_all >= 0) & (ix_all < cw) & (iy_all >= 0) & (iy_all < ch)

    ix_c   = np.clip(ix_all, 0, cw - 1)
    iy_c   = np.clip(iy_all, 0, ch - 1)

    f_val   = pf_crop[iy_c, ix_c].astype(np.float32)
    g_drop  = (grad_x[iy_c, ix_c] * nrm_a[:, 0:1] +
               grad_y[iy_c, ix_c] * nrm_a[:, 1:2]).astype(np.float32)
    inv_brt = (255.0 - img_crop[iy_c, ix_c].astype(np.float32)) / 255.0

    scores  = f_val + 2.0 * g_drop + 1.2 * inv_brt
    scores  = np.where(valid, scores, -1e9)

    step_idx      = np.arange(f_val.shape[1])[None, :]
    norm_dist     = np.abs(steps[None, :]) / float(max(search_range, 1))
    confident     = (f_val > prob_threshold) & valid
    adjusted      = np.where(confident, f_val - distance_weight * norm_dist, -1e9)
    best_conf_idx = np.argmax(adjusted, axis=1)
    has_confident = confident.any(axis=1)

    best_s     = np.argmax(scores, axis=1)
    best_s     = np.where(has_confident, best_conf_idx, best_s)
    best_idx_t = (np.arange(nA), best_s)

    all_oob = ~valid.any(axis=1)
    candidate         = pts_loc.copy()
    best_pts          = np.stack([sx_all[best_idx_t], sy_all[best_idx_t]], axis=1)
    best_pts[all_oob] = pts_a[all_oob]
    candidate[active_arr] = best_pts

    candidate += np.array([xmin_c, ymin_c], dtype=np.float32)
    candidate[:, 0] = gaussian_filter1d(candidate[:, 0], sigma=1.0)
    candidate[:, 1] = gaussian_filter1d(candidate[:, 1], sigma=1.0)
    return [[int(round(np.clip(p[0], 0, w - 1))),
             int(round(np.clip(p[1], 0, h - 1)))] for p in candidate]


def pca_axis_directions(backbone):
    """Calculate the principal axis of a backbone and return outward-pointing vectors for both tips.

    Args:
        backbone (list): Backbone points as [[x, y], ...].

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: (start_dir, end_dir, tip_start, tip_end).
    """
    arr = np.array(backbone, dtype=np.float64)
    tip_start = arr[0]
    tip_end   = arr[-1]

    centered = arr - arr.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    pca_axis = vt[0]

    if np.dot(pca_axis, tip_end - tip_start) < 0:
        pca_axis = -pca_axis

    start_dir = -pca_axis
    end_dir   = pca_axis

    return start_dir, end_dir, tip_start, tip_end

def extend_to_boundary(tip_point, direction, h, w):
    """Walk from tip_point along direction until it intersects the image boundary.

    Args:
        tip_point (tuple[float, float]): Starting point (x, y).
        direction (tuple[float, float]): Direction vector (dx, dy).
        h (int): Full frame height in pixels.
        w (int): Full frame width in pixels.

    Returns:
        np.ndarray: Boundary point [x, y].
    """
    x, y   = tip_point
    dx, dy = direction
    
    t_candidates = []
    
    if abs(dx) > 1e-9:
        t_candidates.append((0 - x) / dx)
        t_candidates.append((w - 1 - x) / dx)
        
    if abs(dy) > 1e-9:
        t_candidates.append((0 - y) / dy)
        t_candidates.append((h - 1 - y) / dy)

    valid_t = [t for t in t_candidates if t > 1e-5]
    
    if not valid_t:
        return np.array([x, y])

    t = min(valid_t)
    final_x = np.clip(x + t * dx, 0, w - 1)
    final_y = np.clip(y + t * dy, 0, h - 1)
    
    return np.array([final_x, final_y])
