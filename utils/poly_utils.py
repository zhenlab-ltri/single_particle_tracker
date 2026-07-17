import cv2
import numpy as np

POLYGON_MODES = {"hole_fill", "hole_crop", "object"}


def sample_contour_evenly(pts, n=120):
    """Resample a closed contour to exactly n evenly-spaced points along its perimeter.

    Args:
        pts (list): Contour points as [[x, y], ...].
        n (int, optional): Number of output points. Defaults to 120.

    Returns:
        np.ndarray: Resampled points of shape (n, 2), dtype float64.
    """
    pts = np.array(pts, dtype=np.float64)
    if len(pts) < 3:
        return np.zeros((n, 2), dtype=np.float64)
    closed   = np.vstack([pts, pts[0]])
    dists    = np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))
    dists    = np.insert(dists, 0, 0)
    total    = dists[-1]
    if total == 0:
        return np.repeat(pts[:1], n, axis=0)
    sample_d = np.linspace(0, total, n, endpoint=False)
    rx = np.interp(sample_d, dists, closed[:, 0])
    ry = np.interp(sample_d, dists, closed[:, 1])
    return np.column_stack([rx, ry])


def enforce_minimum_surface_area(mask, prob_field, initial_area, target_percentage=0.80, prob_threshold=0.2, max_iters=25):
    """Expand a binary mask outward until it reaches a minimum fraction of the initial area.

    Args:
        mask (np.ndarray): Binary mask to expand, dtype uint8.
        prob_field (np.ndarray): Per-pixel probability field of shape (h, w).
        initial_area (int): Reference area in pixels to compute the target from.
        target_percentage (float, optional): Minimum area fraction of initial_area to reach. Defaults to 0.80.
        prob_threshold (float, optional): Minimum probability required to accept an expanded point. Defaults to 0.2.
        max_iters (int, optional): Maximum number of dilation iterations to perform. Defaults to 25.

    Returns:
        np.ndarray: Expanded binary mask, dtype uint8.
    """
    target_area = initial_area * target_percentage
    current = mask.copy()
    area = int(np.sum(current > 0))
    if initial_area <= 0 or area >= target_area:
        return current

    allowed = (prob_field >= prob_threshold).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for _ in range(max_iters):
        grown = cv2.dilate(current, k, iterations=1)
        grown = cv2.bitwise_and(grown, cv2.bitwise_or(allowed, current))
        new_area = int(np.sum(grown > 0))
        if new_area <= area:
            grown = cv2.dilate(current, k, iterations=1)
            new_area = int(np.sum(grown > 0))
        current, area = grown, new_area
        if area >= target_area:
            break
    return current


def optimize_contour_normals(prob_field, reference_pts, target_img_u8, search_range=12):
    """Refine closed-contour points by finding the peak edge-probability along each normal.

    Assumes prob_field comes from boundary-band training (see
    ConformalDeepForestRegressor._boundary_fg_bg): low in the deep interior, peaking at
    the true edge, low again in the deep exterior. A straightforward argmax over this
    shape is safe -- unlike a bulk interior-vs-exterior field, where deep interior values
    form a noisy plateau that can exceed the (comparatively modest) value right at the
    true transition. With boundary-band training, both sides of the true edge are
    explicitly trained low, so the maximum along the ray genuinely corresponds to the
    edge itself.

    Args:
        prob_field (np.ndarray): Per-pixel edge-probability field of shape (h, w).
        reference_pts (list): Contour points as [[x, y], ...].
        target_img_u8 (np.ndarray): Grayscale target frame, dtype uint8. Used as a
            lightweight darkness tiebreaker alongside the probability peak.
        search_range (int, optional): Normal-search radius in pixels. Defaults to 12.

    Returns:
        list[list[int]]: Refined, smoothed points as [[x, y], ...].
    """
    h, w    = prob_field.shape
    pts_arr = np.array(reference_pts, dtype=np.float32)
    n       = len(pts_arr)

    prev_pts = pts_arr[np.arange(n) - 1]
    next_pts = pts_arr[(np.arange(n) + 1) % n]

    tang     = next_pts - prev_pts
    tang_len = np.hypot(tang[:, 0], tang[:, 1]) + 1e-8
    tang    /= tang_len[:, None]

    normals  = np.stack([tang[:, 1], -tang[:, 0]], axis=1)

    cx_poly  = pts_arr[:, 0].mean()
    cy_poly  = pts_arr[:, 1].mean()
    vfc      = pts_arr - np.array([cx_poly, cy_poly], dtype=np.float32)
    flip     = (normals * vfc).sum(axis=1) < 0
    normals[flip] *= -1.0

    steps    = np.arange(-search_range, search_range + 1, dtype=np.float32)

    sx_all   = pts_arr[:, 0:1] + normals[:, 0:1] * steps[None, :]
    sy_all   = pts_arr[:, 1:2] + normals[:, 1:2] * steps[None, :]

    ix_all   = np.round(sx_all).astype(np.int32)
    iy_all   = np.round(sy_all).astype(np.int32)

    valid    = (ix_all >= 0) & (ix_all < w) & (iy_all >= 0) & (iy_all < h)

    ix_c     = np.clip(ix_all, 0, w - 1)
    iy_c     = np.clip(iy_all, 0, h - 1)

    f_val    = prob_field[iy_c, ix_c].astype(np.float32)
    inv_brt  = (255.0 - target_img_u8[iy_c, ix_c].astype(np.float32)) / 255.0

    # Small darkness tiebreaker only -- the probability peak itself is the primary
    # signal now that it genuinely corresponds to the edge, not a noisy plateau.
    scores   = f_val + 0.15 * inv_brt
    scores   = np.where(valid, scores, -1e9)

    best_s   = np.argmax(scores, axis=1)
    best_idx = (np.arange(n), best_s)

    all_oob            = ~valid.any(axis=1)
    candidate          = np.stack([sx_all[best_idx], sy_all[best_idx]], axis=1)
    candidate[all_oob] = pts_arr[all_oob]

    alpha    = 0.35
    prev_c   = candidate[np.arange(n) - 1]
    next_c   = candidate[(np.arange(n) + 1) % n]
    lap      = 0.5 * (prev_c + next_c) - candidate
    smoothed = candidate + alpha * lap

    return [[int(round(np.clip(p[0], 0, w - 1))),
             int(round(np.clip(p[1], 0, h - 1)))] for p in smoothed]


def smooth_closed_contour(pts, iterations=3, alpha=0.5):
    """Couple a closed sequence of points to their neighbors via Laplacian smoothing.

    Each point is pulled partway toward the average of its two neighbors, iterated a
    few times. This is what gives the contour global consistency (a gently-bending,
    non-jagged shape) as an explicit step, rather than relying on independent per-point
    decisions to happen to agree with each other. Unlike a region-based smoothing
    operation, this cannot collapse the shape toward a single point -- it only ever
    locally averages neighboring positions, so the overall size/extent is preserved.

    Args:
        pts (list): Closed contour points as [[x, y], ...].
        iterations (int, optional): Number of smoothing passes. Defaults to 3.
        alpha (float, optional): Blend factor per pass, 0=no change, 1=full replacement
            with the neighbor average. Defaults to 0.5.

    Returns:
        list[list[float]]: Smoothed points as [[x, y], ...].
    """
    p = np.array(pts, dtype=np.float64)
    n = len(p)
    if n < 3:
        return p.tolist()
    for _ in range(iterations):
        prev_p = p[np.arange(n) - 1]
        next_p = p[(np.arange(n) + 1) % n]
        avg    = 0.5 * (prev_p + next_p)
        p      = (1 - alpha) * p + alpha * avg
    return p.tolist()