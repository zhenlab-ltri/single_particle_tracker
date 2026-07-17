import os
import re
import hashlib
import logging
import cv2
import numpy as np

from model.regressor import ConformalDeepForestRegressor

logger = logging.getLogger(__name__)

_gray_cache: dict = {}
_gray_cache_order: list = []
_GRAY_CACHE_MAX = 12

_folder_cache: dict = {}


def make_batches(frames):
    """Partition a list of frames into random-sized batches of 1-5.

    Args:
        frames (list[int]): Frame numbers to partition.

    Returns:
        list[list[int]]: Frames grouped into consecutive batches.
    """
    batches, i = [], 0
    while i < len(frames):
        size = np.random.randint(1, 6)
        batches.append(frames[i:i + size])
        i += size
    return batches


def clear_propagation_caches():
    """Clear all module-level caches between propagation runs.

    Returns:
        None
    """
    _gray_cache.clear()
    _gray_cache_order.clear()
    _folder_cache.clear()


def extract_binary_from_color_image(bgr_img, target_bgr_color):
    """Extract a binary mask from a BGR image by matching a target colour.

    Args:
        bgr_img (np.ndarray): Input image, either BGR (h, w, 3) or single-channel.
        target_bgr_color (list[int]): Target color to match, in BGR order.

    Returns:
        np.ndarray: Binary mask of shape (h, w), dtype uint8.
    """
    if len(bgr_img.shape) == 3 and bgr_img.shape[2] == 3:
        tc   = np.array(target_bgr_color, dtype=np.int16)
        lo   = np.clip(tc - 19, 0,   255).astype(np.uint8)
        hi   = np.clip(tc + 19, 0,   255).astype(np.uint8)
        mask_bin = cv2.inRange(bgr_img, lo, hi)
        if not mask_bin.any():
            gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
            _, mask_bin = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        return mask_bin
    else:
        _, mask_bin = cv2.threshold(bgr_img, 1, 255, cv2.THRESH_BINARY)
        return mask_bin


def find_file_by_frame(folder, frame_num):
    """Locate an image file in a folder whose filename contains the given frame number.

    Args:
        folder (str): Folder to search.
        frame_num (int): Frame number to match against filenames.

    Returns:
        str | None: Full path to the matching file, or None if not found.
    """
    if not folder or not os.path.isdir(folder):
        return None
    if folder not in _folder_cache:
        _folder_cache[folder] = sorted(os.listdir(folder))
    for fn in _folder_cache[folder]:
        if fn.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            match = re.search(r'(\d+)', fn)
            if match and int(match.group(1)) == frame_num:
                return os.path.join(folder, fn)
    return None


def to_gray_u8(frame_data):
    """Convert arbitrary frame data to a normalised uint8 grayscale image.

    Args:
        frame_data (np.ndarray): Input frame, grayscale or multi-channel, any numeric dtype.

    Returns:
        np.ndarray: Normalised grayscale image, dtype uint8.
    """
    content_hash = hashlib.blake2b(np.ascontiguousarray(frame_data).tobytes(),
                                   digest_size=8).digest()
    key = (frame_data.shape, frame_data.dtype.str, content_hash)
    if key in _gray_cache:
        return _gray_cache[key]

    raw = frame_data.astype(np.float32)
    if raw.ndim == 3:
        raw = raw.mean(axis=2)
    mn, mx = raw.min(), raw.max()
    if mx > mn:
        raw = (raw - mn) / (mx - mn)
    result = (raw * 255).astype(np.uint8)

    if len(_gray_cache_order) >= _GRAY_CACHE_MAX:
        oldest = _gray_cache_order.pop(0)
        _gray_cache.pop(oldest, None)
    _gray_cache[key] = result
    _gray_cache_order.append(key)
    return result


def track_one_step_cdum(gsrc, gtgt, gsrc_prev, gtgt_prev, pts, h, w,
                        history=None, edge_mode=None, need_prob=True, anchor_sample=None):
    """Track points from gsrc to gtgt using Lucas-Kanade with forward-backward consistency.

    Args:
        gsrc (np.ndarray): Source grayscale frame, dtype uint8.
        gtgt (np.ndarray): Target grayscale frame, dtype uint8.
        gsrc_prev (np.ndarray | None): Frame before gsrc, used for training. Defaults to None.
        gtgt_prev (np.ndarray | None): Frame before gtgt, used for training. Defaults to None.
        pts (list): Points to track as [[x, y], ...].
        h (int): Full frame height in pixels.
        w (int): Full frame width in pixels.
        history (list, optional): List of (img, prev_img, pts, bbox) tuples used for CDForest training. Defaults to None.
        edge_mode (str | None, optional): 'upper_left', 'lower_right', or None for polygon mode. Defaults to None.
        need_prob (bool, optional): Whether to compute and return a probability field. Defaults to True.
        anchor_sample (tuple, optional): A single (img, prev_img, pts, bbox) tuple treated as
            permanent ground truth (e.g. the original seed frame) and given full,
            undiminished weight in training regardless of the rolling history's decay.
            Defaults to None.

    Returns:
        tuple[list, np.ndarray | None]: Tracked points as [[x, y], ...] and a probability field of shape (h, w), or None if need_prob is False.
    """
    if len(pts) == 0:
        return pts, (np.zeros((h, w), dtype=np.float32) if need_prob else None)

    gsrc_inv = 255 - gsrc
    gtgt_inv = 255 - gtgt

    gsrc_track = cv2.GaussianBlur(gsrc_inv, (7, 7), 0)
    gtgt_track = cv2.GaussianBlur(gtgt_inv, (7, 7), 0)

    ds_factor = 4
    h_ds, w_ds = max(1, h // ds_factor), max(1, w // ds_factor)
    
    gsrc_ds = cv2.resize(gsrc_track, (w_ds, h_ds), interpolation=cv2.INTER_AREA)
    gtgt_ds = cv2.resize(gtgt_track, (w_ds, h_ds), interpolation=cv2.INTER_AREA)

    src_pts    = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
    src_pts_ds = src_pts / ds_factor

    lk_params = dict(winSize=(21, 21), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 35, 0.01))

    p1_ds, st, _   = cv2.calcOpticalFlowPyrLK(gsrc_ds, gtgt_ds, src_pts_ds, None, **lk_params)
    p0b_ds, st2, _ = cv2.calcOpticalFlowPyrLK(gtgt_ds, gsrc_ds, p1_ds,     None, **lk_params)
    
    fb_err = np.linalg.norm(src_pts_ds - p0b_ds, axis=2).flatten()
    good   = (st.flatten() == 1) & (st2.flatten() == 1) & (fb_err < 2.5)

    deltas_ds = (p1_ds - src_pts_ds).reshape(-1, 2)
    rel_ds    = deltas_ds[good] if good.any() else deltas_ds
    dx_ds, dy_ds = float(np.median(rel_ds[:, 0])), float(np.median(rel_ds[:, 1]))

    shifted = []
    for i, (x, y) in enumerate(pts):
        if good[i]:
            nx = float(np.clip(p1_ds[i, 0, 0] * ds_factor, 0, w - 1))
            ny = float(np.clip(p1_ds[i, 0, 1] * ds_factor, 0, h - 1))
        else:
            nx = float(np.clip(x + dx_ds * ds_factor, 0, w - 1))
            ny = float(np.clip(y + dy_ds * ds_factor, 0, h - 1))
        shifted.append([int(round(nx)), int(round(ny))])

    if not need_prob:
        return shifted, None

    arr   = np.array(shifted, dtype=np.float32)
    order = np.argsort(arr[:, 0])
    xs    = arr[order, 0]
    ys    = arr[order, 1]
    xmin  = max(0,     int(np.floor(xs.min())))
    xmax  = min(w,     int(np.ceil( xs.max())) + 1)
    col_range = np.arange(xmin, xmax, dtype=np.float32)
    y_interp  = np.interp(col_range, xs, ys)
    band  = 20
    ymin  = max(0, int(np.floor(y_interp.min())) - band)
    ymax  = min(h, int(np.ceil( y_interp.max())) + band)

    cdf = ConformalDeepForestRegressor(edge_mode=edge_mode)
    ok  = cdf.fit_pipeline(gsrc, gsrc_prev, pts, [ymin, ymax, xmin, xmax],
                           history=history, anchor_sample=anchor_sample)

    if ok:
        # Use the raw, appearance-only field here, not the spatially-modulated one.
        # This prob field feeds normal-direction boundary search downstream
        # (optimize_contour_normals): the spatially-modulated field is highest at the
        # centroid and monotonically decreasing outward BY CONSTRUCTION, so any search
        # that climbs or follows it collapses toward the centroid regardless of where
        # the true membrane actually is. The raw appearance field reflects local
        # brightness/texture evidence only, which is what a genuine edge search needs.
        _, prob = cdf.predict_prob_field(gtgt, gtgt_prev, [ymin, ymax, xmin, xmax],
                                         shifted, h, w)
    else:
        prob = np.zeros((h, w), dtype=np.float32)
        prob[ymin:ymax, xmin:xmax] = (255.0 - gtgt[ymin:ymax, xmin:xmax].astype(np.float32)) / 255.0

    return shifted, prob


def safe_track_step(gsrc, gtgt, gsrc_prev, gtgt_prev, pts, h, w, resample_fn, n_samples,
                    history=None, step_desc="", edge_mode=None, need_prob=True,
                    anchor_sample=None):
    """Run track_one_step_cdum with a last-resort fallback that carries points forward unchanged on failure.

    Args:
        gsrc (np.ndarray): Source grayscale frame, dtype uint8.
        gtgt (np.ndarray): Target grayscale frame, dtype uint8.
        gsrc_prev (np.ndarray | None): Frame before gsrc, used for training.
        gtgt_prev (np.ndarray | None): Frame before gtgt, used for training.
        pts (list): Points to track as [[x, y], ...].
        h (int): Full frame height in pixels.
        w (int): Full frame width in pixels.
        resample_fn (callable): Function(points, n_samples) used to resample tracked or fallback points.
        n_samples (int): Number of points to resample to.
        history (list, optional): History tuples forwarded to track_one_step_cdum. Defaults to None.
        step_desc (str, optional): Description used in the warning log on failure. Defaults to "".
        edge_mode (str | None, optional): 'upper_left', 'lower_right', or None for polygon mode. Defaults to None.
        need_prob (bool, optional): Whether to compute and return a probability field. Defaults to True.
        anchor_sample (tuple, optional): A single (img, prev_img, pts, bbox) tuple treated as
            permanent ground truth and given full, undiminished training weight, forwarded
            to track_one_step_cdum. Defaults to None.

    Returns:
        tuple[list, np.ndarray | None]: Resampled points and a probability field, or None if need_prob is False.
    """
    try:
        tracked, prob = track_one_step_cdum(
            gsrc, gtgt, gsrc_prev, gtgt_prev, pts, h, w, history=history,
            edge_mode=edge_mode, need_prob=need_prob, anchor_sample=anchor_sample)
        return resample_fn(tracked, n_samples), prob
    except Exception:
        logger.warning("safe_track_step: tracking failed (%s), carrying points "
                       "forward unchanged for this step", step_desc, exc_info=True)
        fallback_pts = resample_fn(pts, n_samples)
        return fallback_pts, (np.zeros((h, w), dtype=np.float32) if need_prob else None)


def _smooth_boundary_coordinates(points, window_size=3):
    """Apply a uniform moving-average smooth to the y-coordinates of boundary points.

    Args:
        points (list): Boundary points as [[x, y], ...].
        window_size (int, optional): Moving-average window size, rounded down to odd. Defaults to 3.

    Returns:
        np.ndarray: Points with smoothed y-coordinates, shape (n, 2), dtype float64.
    """
    if len(points) < window_size or window_size < 3:
        return points
    if window_size % 2 == 0:
        window_size -= 1
    pts_copy   = np.array(points, dtype=np.float64)
    y_coords   = pts_copy[:, 1]
    n          = len(y_coords)
    smoothed_y = np.copy(y_coords)
    r          = window_size // 2
    weights    = np.ones(window_size) / window_size
    for i in range(r, n - r):
        smoothed_y[i] = np.dot(weights, y_coords[i - r : i + r + 1])
    pts_copy[:, 1] = smoothed_y
    return pts_copy


def extract_boundary_points(binary_mask, boundary_type):
    """Extract the upper or lower boundary pixel column-by-column from a binary mask.

    Args:
        binary_mask (np.ndarray): Binary mask, dtype uint8.
        boundary_type (str): 'upper' or 'lower', selects which boundary to extract.

    Returns:
        tuple[np.ndarray, float, float]: Boundary points of shape (n, 2), the minimum x, and the maximum x.
    """
    h, w   = binary_mask.shape
    k      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cm     = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, k)
    cm     = cv2.morphologyEx(cm,          cv2.MORPH_OPEN,  k)
    vx     = np.where(np.any(cm > 0, axis=0))[0]
    if len(vx) == 0:
        return np.empty((0, 2)), 0, 0
    if boundary_type == 'upper':
        vy = (h - 1) - np.argmax(cm[::-1] > 0, axis=0)[vx]
    else:
        vy = np.argmax(cm > 0, axis=0)[vx]
    pts = np.column_stack((vx, vy.astype(np.float64)))
    pts = pts[np.argsort(pts[:, 0])]
    pts = _smooth_boundary_coordinates(pts, window_size=3)
    return pts, pts[0, 0], pts[-1, 0]


def execute_mode_geometry(current_mask, pts, edge_mode):
    """Apply geometry operations to a binary mask according to the specified edge mode.

    Args:
        current_mask (np.ndarray): Binary mask to modify, dtype uint8.
        pts (list): Polygon or line points as [[x, y], ...].
        edge_mode (str): One of 'hole_fill', 'hole_crop', 'object', 'upper_left', or 'lower_right'.

    Returns:
        np.ndarray: Resulting binary mask, dtype uint8.
    """
    mat    = current_mask.copy()
    if not pts:
        return mat
    np_pts = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
    h, w   = mat.shape

    if edge_mode == "hole_fill":
        cv2.fillPoly(mat, [np_pts], 255)
        _, final = cv2.threshold(mat, 127, 255, cv2.THRESH_BINARY)
        return final
    elif edge_mode == "hole_crop":
        crop = np.full_like(mat, 255)
        cv2.fillPoly(crop, [np_pts], 0)
        cropped_mat = cv2.bitwise_and(mat, crop)
        _, final = cv2.threshold(cropped_mat, 127, 255, cv2.THRESH_BINARY)
        return final
    elif edge_mode == "object":
        obj = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(obj, [np_pts], 255)
        _, final = cv2.threshold(obj, 127, 255, cv2.THRESH_BINARY)
        return final
    elif edge_mode in ('upper_left', 'lower_right'):
        mask_mat = mat.copy()
        guide = np.zeros((h, w), dtype=np.uint8)
        cv2.polylines(guide, [np.array(pts, dtype=np.int32).reshape(-1, 2)],
                      False, 1, thickness=1)

        guide_cols = np.where(np.any(guide > 0, axis=0))[0]
        if len(guide_cols):
            col_ys = np.array([int(np.mean(np.where(guide[:, x] > 0)[0]))
                               for x in guide_cols], dtype=np.int32)

            row_idx = np.arange(h)
            for x_col, y_line in zip(guide_cols, col_ys):
                if edge_mode == 'upper_left':
                    mask_mat[:y_line, x_col] = 0
                else:
                    mask_mat[y_line + 1:, x_col] = 0

        b_type = 'lower' if edge_mode == 'upper_left' else 'upper'
        mpts, _, _ = extract_boundary_points(mask_mat, b_type)
        region     = np.zeros_like(mask_mat)
        gxs        = guide_cols if len(guide_cols) else np.array([], dtype=int)
        mxs        = mpts[:, 0].astype(int) if len(mpts) else np.array([], dtype=int)

        all_xs = np.unique(np.concatenate([mxs, gxs]))

        m_dict = {}
        if len(mpts):
            for pt in mpts:
                m_dict[int(pt[0])] = int(pt[1])
        g_dict = {}
        for x_col, y_line in zip(guide_cols, col_ys):
            g_dict[int(x_col)] = int(y_line)

        for x in all_xs:
            if x < 0 or x >= w:
                continue
            ym = m_dict.get(x)
            yl = g_dict.get(x)
            if ym is not None and yl is not None:
                region[min(ym, yl):max(ym, yl) + 1, x] = 255
            elif yl is not None:
                if edge_mode == 'upper_left':
                    region[yl:h, x] = 255
                else:
                    region[0:yl + 1, x] = 255

        return cv2.bitwise_or(mask_mat, region)
    return mat
