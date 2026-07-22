import cv2
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from model.cascade_layer import CDForestCascadeLayer
from model.feature_extraction import _extract_multiscale_features


class ConformalDeepForestRegressor:
    """CDForest adapted for pixel-level foreground/background classification in video tracking."""
    MAX_LAYERS = 5

    def __init__(self, edge_mode=None):
        """Initialise an untrained ConformalDeepForestRegressor with an empty layer list.

        Args:
            edge_mode (str | None, optional): None for closed-polygon mode, or a line mode string ('upper_left', 'lower_right') for open-line mode. Defaults to None.

        Returns:
            None
        """
        self.layers      = []
        self.is_trained  = False
        self.edge_mode   = edge_mode

    @staticmethod
    def _clamp_bbox_to_band(pts, bbox, band=20, img_shape=None):
        """Clamp a bounding box to ±band pixels around the pts centroid.

        Args:
            pts (list): Contour or line points as [[x, y], ...].
            bbox (list[int]): Original bounding box [ymin, ymax, xmin, xmax].
            band (int): Half-height/half-width cap in pixels. Defaults to 20.
            img_shape (tuple | None): (h, w) of the source image for boundary clamping.

        Returns:
            list[int]: Clamped bounding box [ymin, ymax, xmin, xmax].
        """
        arr  = np.array(pts, dtype=np.float32)
        if arr.size == 0:
            return bbox
        cy   = float(arr[:, 1].mean())
        cx   = float(arr[:, 0].mean())
        h    = img_shape[0] if img_shape is not None else 1 << 16
        w    = img_shape[1] if img_shape is not None else 1 << 16
        ymin = int(max(0, cy - band))
        ymax = int(min(h, cy + band))
        xmin = int(max(0, cx - band))
        xmax = int(min(w, cx + band))
        ymin = max(ymin, bbox[0])
        ymax = min(ymax, bbox[1])
        xmin = max(xmin, bbox[2])
        xmax = min(xmax, bbox[3])
        if ymax - ymin < 4:
            ymin, ymax = bbox[0], bbox[1]
        if xmax - xmin < 4:
            xmin, xmax = bbox[2], bbox[3]
        return [ymin, ymax, xmin, xmax]

    def _boundary_fg_bg(self, pts, h_roi, w_roi, img_roi=None, band=4):
        """Generate boundary as foreground and interior+exterior background masks.

        Args:
            pts (list): Polygon vertices as [[x, y], ...] in ROI-local coordinates.
            h_roi (int): Height of the ROI patch in pixels.
            w_roi (int): Width of the ROI patch in pixels.
            img_roi (np.ndarray | None, optional): Grayscale ROI patch, dtype uint8.
            band (int, optional): Half-width in pixels of the foreground boundary ring.
                Defaults to 4.

        Returns:
            tuple[np.ndarray, np.ndarray]: (fg, bg) each of shape (h_roi, w_roi), dtype uint8.
        """
        filled = np.zeros((h_roi, w_roi), dtype=np.uint8)
        cv2.fillPoly(filled, [np.array(pts, dtype=np.int32).reshape(-1, 1, 2)], 1)

        k_band     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1,) * 2)
        outer_ring = cv2.dilate(filled, k_band, iterations=1)
        inner_ring = cv2.erode(filled, k_band, iterations=1)
        fg = np.logical_and(outer_ring == 1, inner_ring == 0).astype(np.uint8)

        deep_in_k     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (band + 3) + 1,) * 2)
        deep_interior = cv2.erode(filled, deep_in_k, iterations=1)

        deep_out_k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (band + 6) + 1,) * 2)
        deep_dilated  = cv2.dilate(filled, deep_out_k, iterations=1)
        deep_exterior = (deep_dilated == 0).astype(np.uint8)

        bg = np.logical_or(deep_interior == 1, deep_exterior == 1).astype(np.uint8)

        if int(fg.sum()) < 8:
            fg = np.logical_and(outer_ring == 1, filled == 0).astype(np.uint8)

        return fg, bg

    def _line_fg_bg(self, pts, h_roi, w_roi, img_roi, margin=3, band=10):
        """Generate foreground/background training bands straddling an open line, using brightness to decide sides.

        Args:
            pts (list): Line points as [[x, y], ...] in ROI-local coordinates.
            h_roi (int): Height of the ROI patch in pixels.
            w_roi (int): Width of the ROI patch in pixels.
            img_roi (np.ndarray): Grayscale ROI patch, dtype uint8.
            margin (int, optional): Gap in pixels left empty on each side of the line. Defaults to 3.
            band (int, optional): Width in pixels of each side's candidate band. Defaults to 10.

        Returns:
            tuple[np.ndarray, np.ndarray]: (fg_mask, bg_mask) each of shape (h_roi, w_roi), dtype uint8.
        """
        empty = np.zeros((h_roi, w_roi), dtype=np.uint8)
        if len(pts) < 2:
            return empty, empty.copy()

        guide = np.zeros((h_roi, w_roi), dtype=np.uint8)
        line_pts = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(guide, [line_pts], False, 1, thickness=1)

        guide_cols = np.where(np.any(guide > 0, axis=0))[0]
        if len(guide_cols) == 0:
            return empty, empty.copy()

        col_ys = np.array([int(np.mean(np.where(guide[:, x] > 0)[0]))
                           for x in guide_cols], dtype=np.int32)

        side_above = np.zeros((h_roi, w_roi), dtype=np.uint8)
        side_below = np.zeros((h_roi, w_roi), dtype=np.uint8)
        for x_col, y_line in zip(guide_cols, col_ys):
            a0, a1 = max(0, y_line - margin - band), max(0, y_line - margin)
            if a1 > a0:
                side_above[a0:a1, x_col] = 1
            b0, b1 = min(h_roi, y_line + margin), min(h_roi, y_line + margin + band)
            if b1 > b0:
                side_below[b0:b1, x_col] = 1

        above_px = img_roi[side_above == 1]
        below_px = img_roi[side_below == 1]
        if above_px.size == 0 or below_px.size == 0:
            return empty, empty.copy()

        if above_px.mean() <= below_px.mean():
            return side_above, side_below
        return side_below, side_above

    def _extract_frame_samples(self, img_u8, prev_img_u8, pts, bbox, max_fg=None):
        """Extract labelled pixel samples from one frame for training.

        Args:
            img_u8 (np.ndarray): Current frame as a uint8 grayscale image.
            prev_img_u8 (np.ndarray): Previous frame, or None if unavailable.
            pts (list): Polygon or contour vertices as [[x, y], ...].
            bbox (list[int]): Bounding box [ymin, ymax, xmin, xmax].
            max_fg (int, optional): Cap on foreground samples. Defaults to None.

        Returns:
            tuple[np.ndarray, np.ndarray] | tuple[None, None]: (X, y) arrays or (None, None) if ROI is too small.
        """
        ymin, ymax, xmin, xmax = bbox
        h_roi = ymax - ymin;  w_roi = xmax - xmin
        if h_roi < 4 or w_roi < 4:
            return None, None

        pts_roi             = [[x - xmin, y - ymin] for x, y in pts]
        img_roi              = img_u8[ymin:ymax, xmin:xmax]
        if self.edge_mode is not None:
            fg_mask, bg_mask = self._line_fg_bg(pts_roi, h_roi, w_roi, img_roi)
        else:
            fg_mask, bg_mask = self._boundary_fg_bg(pts_roi, h_roi, w_roi, img_roi)
        fg_idx              = np.where(fg_mask.flatten() == 1)[0]
        bg_idx              = np.where(bg_mask.flatten() == 1)[0]

        if len(fg_idx) == 0 or len(bg_idx) == 0:
            return None, None

        n_fg = min(len(fg_idx), max_fg) if max_fg else len(fg_idx)
        fg_s = np.random.choice(fg_idx, n_fg, replace=False)
        n_bg = min(len(bg_idx), n_fg * 2)
        bg_s = np.random.choice(bg_idx, n_bg, replace=False)
        sel  = np.concatenate([fg_s, bg_s])
        y_lbl = np.concatenate([np.ones(n_fg), np.zeros(n_bg)])

        X_full = _extract_multiscale_features(img_u8, prev_img_u8,
                                              ymin, ymax, xmin, xmax)
        return X_full[sel], y_lbl

    def fit_pipeline(self, img_src_u8, prev_img_src_u8, pts_src, bounding_box,
                     history=None, anchor_sample=None):
        """Build the CDForest cascade, auto-determining depth by validation AUC.

        Args:
            img_src_u8 (np.ndarray): Source frame as a uint8 grayscale image.
            prev_img_src_u8 (np.ndarray): Previous source frame, or None if unavailable.
            pts_src (list): Contour or polygon points as [[x, y], ...].
            bounding_box (list[int]): Bounding box [ymin, ymax, xmin, xmax].
            history (list, optional): List of (img, prev_img, pts, bbox) tuples, oldest to
                newest, contributing with decaying weight the further back they are.
                Defaults to None.
            anchor_sample (tuple, optional): A single (img, prev_img, pts, bbox) tuple
                treated as permanent ground truth and given a full, undiminished sample budget regardless of how
                much rolling history has accumulated. Without this, any anchor placed in the ordinary rolling
                history eventually gets discounted down. Defaults to None.

        Returns:
            bool: True if at least one cascade layer was successfully trained, False otherwise.
        """
        X_cur, y_cur = self._extract_frame_samples(
            img_src_u8, prev_img_src_u8, pts_src, bounding_box)
        if X_cur is None:
            return False

        all_X, all_y = [X_cur], [y_cur]
        n_fg_cur = int((y_cur == 1).sum())

        if anchor_sample is not None:
            a_img, a_prev, a_pts, a_bbox = anchor_sample
            a_bbox = self._clamp_bbox_to_band(a_pts, a_bbox, band=20, img_shape=a_img.shape)
            Xa, ya = self._extract_frame_samples(a_img, a_prev, a_pts, a_bbox,
                                                 max_fg=max(n_fg_cur, 4))
            if Xa is not None:
                all_X.append(Xa); all_y.append(ya)

        if history:
            decay    = 0.6
            for k, (h_img, h_prev, h_pts, h_bbox) in enumerate(reversed(history)):
                max_fg = max(4, int(n_fg_cur * (decay ** (k + 1))))

                h_bbox = self._clamp_bbox_to_band(h_pts, h_bbox, band=20,
                                                  img_shape=h_img.shape)

                Xh, yh = self._extract_frame_samples(h_img, h_prev, h_pts, h_bbox,
                                                     max_fg=max_fg)
                if Xh is not None:
                    all_X.append(Xh); all_y.append(yh)

        X_base = np.vstack(all_X)
        y_lbl  = np.concatenate(all_y)

        try:
            X_tr, X_val_stop, y_tr, y_val_stop = train_test_split(
                X_base, y_lbl, test_size=0.2, stratify=y_lbl, random_state=7)
        except Exception:
            X_tr, X_val_stop, y_tr, y_val_stop = train_test_split(
                X_base, y_lbl, test_size=0.2, random_state=7)

        self.layers = []
        X_cascade   = X_tr.copy()
        X_val_casc  = X_val_stop.copy()
        best_val_score = -1.0

        for ell in range(self.MAX_LAYERS):
            n_est = 50 if ell == 0 else 20
            layer = CDForestCascadeLayer(n_estimators=n_est)
            ok    = layer.fit_and_calibrate(X_cascade, y_tr)
            if not ok:
                break

            aug_tr  = layer.transform_and_filter(X_cascade)
            aug_val = layer.transform_and_filter(X_val_casc)

            X_cascade_next  = np.hstack([X_tr, aug_tr])
            X_val_casc_next = np.hstack([X_val_stop, aug_val])

            w_fg  = layer.inference_weights(y_label=1)
            scores = np.zeros(len(X_val_casc), dtype=np.float32)
            for t, est in enumerate(layer.estimators):
                p      = layer._safe_proba(est, X_val_casc)
                scores += p[:, 1] * w_fg[t]

            try:
                val_auc = roc_auc_score(y_val_stop, scores)
            except Exception:
                val_auc = 0.5

            self.layers.append(layer)
            X_cascade  = X_cascade_next
            X_val_casc = X_val_casc_next

            if val_auc <= best_val_score:
                break
            best_val_score = val_auc

        if not self.layers:
            return False

        self.is_trained = True
        return True

    def predict_prob_field(self, img_tgt_u8, prev_img_tgt_u8, bounding_box,
                           prior_pts, h, w, spatial_sigma_factor=0.85):
        """Produce per-pixel foreground probability fields over the full frame.

        Args:
            img_tgt_u8 (np.ndarray): Target frame as a uint8 grayscale image.
            prev_img_tgt_u8 (np.ndarray): Previous target frame, or None if unavailable.
            bounding_box (list[int]): Bounding box [ymin, ymax, xmin, xmax].
            prior_pts (list): Tracked contour points as [[x, y], ...].
            h (int): Full frame height in pixels.
            w (int): Full frame width in pixels.
            spatial_sigma_factor (float, optional): Multiplier on the object's own estimated
                radius. Defaults to 0.85

        Returns:
            tuple[np.ndarray, np.ndarray]: (prob_field, raw_appearance_field), both shape
            (h, w), dtype float32. prob_field is the appearance score multiplied by a
            spatial Gaussian centered on the current centroid.
            raw_appearance_field is the classifier's appearance score with no spatial
            term applied.
        """
        prob_field       = np.zeros((h, w), dtype=np.float32)
        raw_appearance   = np.zeros((h, w), dtype=np.float32)
        if not self.is_trained or not self.layers:
            return prob_field, raw_appearance

        ymin, ymax, xmin, xmax = bounding_box
        X_full = _extract_multiscale_features(img_tgt_u8, prev_img_tgt_u8,
                                              ymin, ymax, xmin, xmax)

        X_ext = X_full.copy()
        for layer in self.layers[:-1]:
            aug   = layer.transform_and_filter(X_ext)
            X_ext = np.hstack([X_full, aug])

        final_layer = self.layers[-1]
        rng         = np.random.default_rng(seed=42)

        pvals = final_layer.conformal_pvalues(X_ext, rng)

        w_fg = final_layer.inference_weights(y_label=1)
        w_bg = final_layer.inference_weights(y_label=0)

        agg_p_fg = (pvals[:, :, 1] * w_fg[None, :]).sum(axis=1)
        agg_p_bg = (pvals[:, :, 0] * w_bg[None, :]).sum(axis=1)

        eps         = final_layer.epsilon
        in_set_fg   = (agg_p_fg > eps).astype(np.float32)
        in_set_bg   = (agg_p_bg > eps).astype(np.float32)

        fg_prob = agg_p_fg * (0.7 + 0.3 * in_set_fg)

        p_max = fg_prob.max()
        if p_max > 0:
            fg_prob = fg_prob / p_max

        roi_h, roi_w = ymax - ymin, xmax - xmin
        fg_prob_2d   = fg_prob.reshape(roi_h, roi_w)
        raw_appearance[ymin:ymax, xmin:xmax] = fg_prob_2d

        prior_arr   = np.array(prior_pts, dtype=np.float32)
        cx, cy      = prior_arr[:, 0].mean(), prior_arr[:, 1].mean()
        r_est       = float(np.max(np.linalg.norm(prior_arr - [cx, cy], axis=1))) + 1.0
        yy, xx      = np.mgrid[ymin:ymax, xmin:xmax]
        spatial     = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) /
                              (2.0 * (r_est * spatial_sigma_factor) ** 2)).astype(np.float32)

        prob_field[ymin:ymax, xmin:xmax] = fg_prob_2d * spatial
        return prob_field, raw_appearance
