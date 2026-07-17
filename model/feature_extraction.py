import hashlib
import cv2
import numpy as np

_feat_cache: dict = {}
_feat_cache_order: list = []
_FEAT_CACHE_MAX = 20


def _roi_key(img_u8, prev_img_u8, ymin, ymax, xmin, xmax):
    """Build a stable cache key from a content hash of the ROI and previous-frame ROI.

    Args:
        img_u8 (np.ndarray): Current frame as a uint8 grayscale image.
        prev_img_u8 (np.ndarray | None): Previous frame as a uint8 grayscale image, or None if unavailable.
        ymin (int): Top row of the ROI bounding box.
        ymax (int): Bottom row of the ROI bounding box.
        xmin (int): Left column of the ROI bounding box.
        xmax (int): Right column of the ROI bounding box.

    Returns:
        tuple: Cache key encoding ROI shape, dtype, and content hashes.
    """
    roi = img_u8[ymin:ymax, xmin:xmax]
    roi_hash = hashlib.blake2b(np.ascontiguousarray(roi).tobytes(), digest_size=8).digest()

    if prev_img_u8 is not None:
        p_roi = prev_img_u8[ymin:ymax, xmin:xmax]
        prev_hash = hashlib.blake2b(np.ascontiguousarray(p_roi).tobytes(), digest_size=8).digest()
    else:
        prev_hash = None

    k = (roi.shape, img_u8.dtype.str, roi_hash, prev_hash)
    return k


def clear_feature_cache():
    """Clear the multiscale feature cache between propagation runs.

    Returns:
        None
    """
    _feat_cache.clear()
    _feat_cache_order.clear()


def extract_pixel_features_roi(img_u8, prev_img_u8, ymin, ymax, xmin, xmax):
    """Extract per-pixel features from a bounded region-of-interest (ROI) crop.

    Args:
        img_u8 (np.ndarray): Current frame as a uint8 grayscale image.
        prev_img_u8 (np.ndarray): Previous frame as a uint8 grayscale image, or None if unavailable.
        ymin (int): Top row of the ROI bounding box (inclusive).
        ymax (int): Bottom row of the ROI bounding box (exclusive).
        xmin (int): Left column of the ROI bounding box (inclusive).
        xmax (int): Right column of the ROI bounding box (exclusive).

    Returns:
        np.ndarray: Feature matrix of shape (n_pixels, n_features) where n_pixels = (ymax-ymin) * (xmax-xmin), dtype float32.
    """
    roi   = img_u8[ymin:ymax, xmin:xmax]
    p_roi = prev_img_u8[ymin:ymax, xmin:xmax] if prev_img_u8 is not None else None
    return extract_pixel_features_with_flow(roi, p_roi)


def extract_pixel_features_with_flow(img_u8, prev_img_u8=None):
    """Extract per-pixel features from an ROI patch using inverted image as the primary signal.

    Args:
        img_u8 (np.ndarray): Current ROI patch as a uint8 grayscale image.
        prev_img_u8 (np.ndarray, optional): Previous ROI patch as a uint8 grayscale image. Defaults to None.

    Returns:
        np.ndarray: Feature matrix of shape (n_pixels, n_features), dtype float32.
    """
    img            = (255.0 - img_u8.astype(np.float32))
    img_native_raw = img_u8.astype(np.float32)

    feats = [img.flatten(), img_native_raw.flatten()]

    for sigma in (1.0, 3.0):
        blur_tgt = cv2.GaussianBlur(img,            (0, 0), sigmaX=sigma)
        blur_bak = cv2.GaussianBlur(img_native_raw, (0, 0), sigmaX=sigma)
        feats.append(blur_tgt.flatten())
        feats.append(blur_bak.flatten())

        gx_t = cv2.Sobel(blur_tgt, cv2.CV_32F, 1, 0, ksize=3)
        gy_t = cv2.Sobel(blur_tgt, cv2.CV_32F, 0, 1, ksize=3)
        feats.append(cv2.magnitude(gx_t, gy_t).flatten())

        gx_b = cv2.Sobel(blur_bak, cv2.CV_32F, 1, 0, ksize=3)
        gy_b = cv2.Sobel(blur_bak, cv2.CV_32F, 0, 1, ksize=3)
        feats.append(cv2.magnitude(gx_b, gy_b).flatten())

        Ixx  = cv2.GaussianBlur(gx_t * gx_t, (0, 0), sigmaX=sigma)
        Iyy  = cv2.GaussianBlur(gy_t * gy_t, (0, 0), sigmaX=sigma)
        Ixy  = cv2.GaussianBlur(gx_t * gy_t, (0, 0), sigmaX=sigma)
        tr   = Ixx + Iyy
        det  = Ixx * Iyy - Ixy * Ixy
        disc = np.sqrt(np.maximum(0.0, tr * tr / 4.0 - det))
        feats.append((tr / 2.0 + disc).flatten())
        feats.append((tr / 2.0 - disc).flatten())

    if prev_img_u8 is not None:
        flow  = cv2.calcOpticalFlowFarneback(
            255 - prev_img_u8, 255 - img_u8,
            None, 0.5, 3, 15, 3, 5, 1.2, 0)
        feats.append((flow[:, :, 0].flatten() * 0.15))
        feats.append((flow[:, :, 1].flatten() * 0.15))
    else:
        feats.append(np.zeros(img.size, dtype=np.float32))
        feats.append(np.zeros(img.size, dtype=np.float32))

    return np.column_stack(feats).astype(np.float32)


def _extract_multiscale_features(img_u8, prev_img_u8, ymin, ymax, xmin, xmax):
    """Extract features at three spatial scales (full, half, quarter) and concatenate them.

    Args:
        img_u8 (np.ndarray): Current frame as a uint8 grayscale image.
        prev_img_u8 (np.ndarray): Previous frame as a uint8 grayscale image, or None if unavailable.
        ymin (int): Top row of the ROI bounding box.
        ymax (int): Bottom row of the ROI bounding box.
        xmin (int): Left column of the ROI bounding box.
        xmax (int): Right column of the ROI bounding box.

    Returns:
        np.ndarray: Concatenated feature matrix of shape (n_pixels, 3 * n_features), dtype float32.
    """
    cache_key = _roi_key(img_u8, prev_img_u8, ymin, ymax, xmin, xmax)
    if cache_key in _feat_cache:
        return _feat_cache[cache_key]
    h_roi = ymax - ymin
    w_roi = xmax - xmin

    X1 = extract_pixel_features_roi(img_u8, prev_img_u8, ymin, ymax, xmin, xmax)

    def _downsample_features(img_u8, prev_img_u8, ymin, ymax, xmin, xmax, scale):
        """Extract features at 1/scale resolution then upsample back to the full ROI grid.

        Args:
            img_u8 (np.ndarray): Current frame as a uint8 grayscale image.
            prev_img_u8 (np.ndarray): Previous frame as a uint8 grayscale image, or None if unavailable.
            ymin (int): Top row of the full-resolution ROI.
            ymax (int): Bottom row of the full-resolution ROI.
            xmin (int): Left column of the full-resolution ROI.
            xmax (int): Right column of the full-resolution ROI.
            scale (int): Downsampling factor (e.g. 2 = half size, 4 = quarter size).

        Returns:
            np.ndarray: Feature matrix upsampled back to the full ROI spatial grid, shape (n_pixels, n_features), dtype float32.
        """
        h, w = img_u8.shape[:2]
        cy   = (ymin + ymax) // 2
        cx   = (xmin + xmax) // 2
        half_h = max(4, (ymax - ymin) // (2 * scale))
        half_w = max(4, (xmax - xmin) // (2 * scale))
        y0 = max(0, cy - half_h);  y1 = min(h, cy + half_h)
        x0 = max(0, cx - half_w);  x1 = min(w, cx + half_w)

        roi_s      = img_u8[y0:y1, x0:x1]
        p_roi_s    = prev_img_u8[y0:y1, x0:x1] if prev_img_u8 is not None else None
        roi_full   = cv2.resize(roi_s,   (w_roi, h_roi), interpolation=cv2.INTER_LINEAR)
        p_roi_full = (cv2.resize(p_roi_s, (w_roi, h_roi), interpolation=cv2.INTER_LINEAR)
                      if p_roi_s is not None else None)
        return extract_pixel_features_with_flow(roi_full, p_roi_full)

    X2 = _downsample_features(img_u8, prev_img_u8, ymin, ymax, xmin, xmax, scale=2)
    X3 = _downsample_features(img_u8, prev_img_u8, ymin, ymax, xmin, xmax, scale=4)

    result = np.hstack([X1, X2, X3])

    if len(_feat_cache_order) >= _FEAT_CACHE_MAX:
        oldest = _feat_cache_order.pop(0)
        _feat_cache.pop(oldest, None)
    _feat_cache[cache_key] = result
    _feat_cache_order.append(cache_key)

    return result
