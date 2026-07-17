import os
import logging
import h5py
import cv2
import numpy as np


logger = logging.getLogger(__name__)

from utils.shared_utils import (
    find_file_by_frame,
    to_gray_u8,
    extract_binary_from_color_image,
    execute_mode_geometry,
    safe_track_step,
    clear_propagation_caches,
    make_batches
)
from utils.edge_utils import (
    sample_line_evenly,
    optimize_line_normals,
    pca_axis_directions,
    extend_to_boundary,
)
from model.regressor import ConformalDeepForestRegressor


def _line_band_bbox(pts, h, w, band=20):
    """Compute a bounding box around a line's interpolated y-band across its x-span.

    Args:
        pts (list): Line points as [[x, y], ...].
        h (int): Full frame height in pixels.
        w (int): Full frame width in pixels.
        band (int, optional): Extra padding in pixels added above/below the
            interpolated band. Defaults to 20.

    Returns:
        list[int]: Bounding box [ymin, ymax, xmin, xmax].
    """
    arr   = np.array(pts, dtype=np.float32)
    arr   = arr[np.argsort(arr[:, 0])]
    xs, ys = arr[:, 0], arr[:, 1]
    x0        = max(0,     int(np.floor(xs.min())))
    x1        = min(w - 1, int(np.ceil(xs.max())))
    col_range = np.arange(x0, x1 + 1, dtype=np.float32)
    y_interp  = np.interp(col_range, xs, ys)
    ymin      = max(0, int(np.floor(y_interp.min())) - band)
    ymax      = min(h, int(np.ceil(y_interp.max())) + band)
    return [ymin, ymax, x0, x1 + 1]


def _line_prob_field(gray_u8, prev_gray_u8, pts, edge_mode, h, w, history=None):
    """Train/predict a CDForest probability field for a line against a single frame.

    Args:
        gray_u8 (np.ndarray): Grayscale frame to both train on and predict over.
        prev_gray_u8 (np.ndarray | None): Previous frame, or None if unavailable.
        pts (list): Line points as [[x, y], ...].
        edge_mode (str): 'upper_left' or 'lower_right'.
        h (int): Full frame height in pixels.
        w (int): Full frame width in pixels.
        history (list, optional): Same history format as track_one_step_cdum. Defaults to None.

    Returns:
        np.ndarray | None: Probability field of shape (h, w), or None if training failed.
    """
    arr = np.array(pts, dtype=np.float32)
    pad = 20
    bbox = [max(0, int(arr[:, 1].min()) - pad), min(h, int(arr[:, 1].max()) + pad),
            max(0, int(arr[:, 0].min()) - pad), min(w, int(arr[:, 0].max()) + pad)]
    try:
        cdf = ConformalDeepForestRegressor(edge_mode=edge_mode)
        ok  = cdf.fit_pipeline(gray_u8, prev_gray_u8, pts, bbox, history=history)
        if ok:
            # predict_prob_field now returns (prob_field, raw_appearance); line-mode
            # tracking previously received only the spatially-modulated field, so take
            # that one specifically to preserve identical prior behavior here.
            prob_field, _ = cdf.predict_prob_field(gray_u8, prev_gray_u8, bbox, pts, h, w)
            return prob_field
    except Exception:
        logger.warning("_line_prob_field: training failed, caller will fall", exc_info=True)
    return None

def _edge_setup(video_data, frame_range, flat_pts, edge_mode,
                input_folder, output_folder, target_color, n_samples, search_range):
    """Process the start frame of an edge-stroke propagation run.

    Args:
        video_data (h5py.Dataset): Video frames indexed by frame number.
        frame_range (list[int]): Frames to process, starting with the seed frame.
        flat_pts (list): Initial line points as [[x, y], ...].
        edge_mode (str): 'upper_left' or 'lower_right'.
        input_folder (str): Folder containing source mask images.
        output_folder (str): Folder to write propagated mask images to.
        target_color (list[int]): Target BGR color identifying the mask region.
        n_samples (int): Number of points to resample the line to.
        search_range (int): Normal-search radius in pixels for refinement.

    Returns:
        dict: State dict carrying frame data, timeline, and tracking context for later batches.
    """
    h, w  = video_data[frame_range[0]].shape[:2]
    N     = n_samples

    seed_pts = sample_line_evenly(flat_pts, N)
    timeline = {}
    prev_line = None

    start_f  = frame_range[0]
    start_tp = find_file_by_frame(output_folder, start_f) or find_file_by_frame(input_folder, start_f)
    if start_tp:
        start_img = cv2.imread(start_tp, cv2.IMREAD_COLOR)
        if start_img is not None:
            start_gray = cv2.cvtColor(start_img, cv2.COLOR_BGR2GRAY)
            inv_start  = (255 - start_gray).astype(np.float32) / 255.0

            full_seed  = np.vstack([seed_pts])
            final_seed = sample_line_evenly(full_seed, N)

            start_field = _line_prob_field(start_gray, None, final_seed.tolist(),
                                           edge_mode, h, w)
            if start_field is None or start_field.max() <= 0:
                start_field = inv_start

            refined    = optimize_line_normals(start_field, final_seed.tolist(), start_gray,
                                               edge_mode, search_range=search_range,
                                               active_indices=np.arange(len(final_seed)))
            final_seed = np.array(refined, dtype=np.float64)

            start_bin  = extract_binary_from_color_image(start_img, target_color)
            start_proc = execute_mode_geometry(start_bin, flat_pts, edge_mode)

            cm = np.zeros((h, w, 3), dtype=np.uint8)
            cm[start_proc > 0] = [target_color[2], target_color[1], target_color[0]]
            cv2.imwrite(start_tp, cm)

            timeline[start_f] = [{"type": "point", "pts": final_seed.tolist(), "edge_mode": edge_mode}]
            prev_line = final_seed.copy()

    start_tl = {start_f: timeline[start_f]} if start_f in timeline else {}
    return dict(
        video_data=video_data, frame_range=frame_range, edge_mode=edge_mode,
        input_folder=input_folder, output_folder=output_folder,
        target_color=target_color, n_samples=N, search_range=search_range,
        h=h, w=w, timeline=timeline,
        prev_line=prev_line, start_tl=start_tl,
    )

def _edge_batch(state, batch, stop_event=None):
    """Track a batch of frames forward and backward from the anchor frame and blend the results.

    Args:
        state (dict): State dict produced by _edge_setup or a prior _edge_batch call.
        batch (list[int]): Frame numbers to process in this batch.
        stop_event (threading.Event, optional): Event used to signal early stop. Defaults to None.

    Returns:
        dict: Timeline entries for the frames processed in this batch, keyed by frame number.
    """
    video_data    = state['video_data']
    frame_range   = state['frame_range']
    edge_mode     = state['edge_mode']
    input_folder  = state['input_folder']
    output_folder = state['output_folder']
    target_color  = state['target_color']
    N             = state['n_samples']
    search_range  = state['search_range']
    h, w          = state['h'], state['w']
    prev_line     = state['prev_line']

    anchor      = frame_range[frame_range.index(batch[0]) - 1]
    batch_range = [anchor] + list(batch)
    total_b     = len(batch_range)

    anchor_pts = state['timeline'][anchor][0]['pts']

    local_fwd, local_hist_fwd = {batch_range[0]: anchor_pts}, []
    local_prob_fwd = {}
    
    g_anchor = to_gray_u8(video_data[anchor])
    safe_track_step(
        g_anchor, g_anchor, None, g_anchor, anchor_pts, h, w,
        sample_line_evenly, N, history=local_hist_fwd,
        step_desc="edge warm start fwd", edge_mode=edge_mode,
        need_prob=True)
    local_hist_fwd.append((g_anchor, None, anchor_pts, _line_band_bbox(anchor_pts, h, w)))

    for idx in range(total_b - 1):
        fc, fn    = batch_range[idx], batch_range[idx + 1]
        gsrc      = to_gray_u8(video_data[fc])
        gtgt      = to_gray_u8(video_data[fn])
        gsrc_prev = to_gray_u8(video_data[batch_range[idx - 1]]) if idx > 0 else g_anchor
        
        tracked, prob = safe_track_step(
            gsrc, gtgt, gsrc_prev, gsrc, local_fwd[fc], h, w,
            sample_line_evenly, N, history=local_hist_fwd,
            step_desc=f"local fwd fc={fc}->fn={fn}", edge_mode=edge_mode,
            need_prob=True)
        local_hist_fwd.append((gsrc, gsrc_prev, local_fwd[fc], _line_band_bbox(tracked, h, w)))
        if len(local_hist_fwd) > 5:
            local_hist_fwd.pop(0)
        local_fwd[fn] = tracked
        local_prob_fwd[fn] = prob

    local_bwd, local_hist_bwd = {batch_range[-1]: local_fwd[batch_range[-1]]}, []
    local_prob_bwd = {}
    
    anchor_bwd_pts = local_fwd[batch_range[-1]]
    g_anchor_bwd = to_gray_u8(video_data[batch_range[-1]])
    safe_track_step(
        g_anchor_bwd, g_anchor_bwd, None, g_anchor_bwd, anchor_bwd_pts, h, w,
        sample_line_evenly, N, history=local_hist_bwd,
        step_desc="edge warm start bwd", edge_mode=edge_mode,
        need_prob=True)
    local_hist_bwd.append((g_anchor_bwd, None, anchor_bwd_pts, _line_band_bbox(anchor_bwd_pts, h, w)))

    for idx in range(total_b - 1, 0, -1):
        fc, fp    = batch_range[idx], batch_range[idx - 1]
        gsrc      = to_gray_u8(video_data[fc])
        gtgt      = to_gray_u8(video_data[fp])
        gsrc_prev = to_gray_u8(video_data[batch_range[idx + 1]]) if idx < total_b - 1 else g_anchor_bwd
        
        tracked, prob = safe_track_step(
            gsrc, gtgt, gsrc_prev, gsrc, local_bwd[fc], h, w,
            sample_line_evenly, N, history=local_hist_bwd,
            step_desc=f"local bwd fc={fc}->fp={fp}", edge_mode=edge_mode,
            need_prob=True)
        local_hist_bwd.append((gsrc, gsrc_prev, local_bwd[fc], _line_band_bbox(tracked, h, w)))
        if len(local_hist_bwd) > 5:
            local_hist_bwd.pop(0)
        local_bwd[fp] = tracked
        local_prob_bwd[fp] = prob

    batch_tl = {}
    for f in batch:
        if stop_event is not None and stop_event.is_set():
            break
        tp = find_file_by_frame(output_folder, f) or find_file_by_frame(input_folder, f)
        if not tp:
            continue
        raw_img = cv2.imread(tp, cv2.IMREAD_COLOR)
        if raw_img is None:
            continue
        gtgt_gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        inv_gray  = (255 - gtgt_gray).astype(np.float32) / 255.0

        pf_f = local_prob_fwd.get(f)
        pf_b = local_prob_bwd.get(f)
        if pf_f is not None and pf_b is not None:
            line_field = np.maximum(pf_f, pf_b)
        elif pf_f is not None:
            line_field = pf_f
        elif pf_b is not None:
            line_field = pf_b
        else:
            line_field = inv_gray

        lf = np.array(local_fwd[f], dtype=np.float64)
        lb = np.array(local_bwd[f], dtype=np.float64)
        
        backbone = 0.5 * (lf + lb)
        backbone = sample_line_evenly(backbone, N)

        if prev_line is not None:
            disp        = np.linalg.norm(backbone - prev_line, axis=1)
            alpha       = float(np.clip(0.55 - (float(np.mean(disp)) / 30.0), 0.40, 0.65))
            backbone    = alpha * backbone + (1 - alpha) * prev_line
            refine_mask = disp > 0.1
        else:
            refine_mask = np.ones(len(backbone), dtype=bool)

        prev_line = backbone.copy()

        ri = np.where(refine_mask)[0]
        if len(ri) > 0:
            try:
                refined  = optimize_line_normals(line_field, backbone.tolist(), gtgt_gray,
                                                 edge_mode, search_range=search_range,
                                                 active_indices=ri)
                backbone = backbone.copy()
                backbone[ri] = np.array(refined, dtype=np.float64)[ri]
            except Exception:
                logger.warning("edge batch: normal-refinement failed for frame %d, "
                               "using unrefined tracked line", f, exc_info=True)

        try:
            s_dir, e_dir, tip_s, tip_e = pca_axis_directions(backbone)
            full_pts   = np.vstack([ 
                                    extend_to_boundary(tip_s, s_dir, h, w).reshape(1, 2),
                                    backbone,
                                    extend_to_boundary(tip_e, e_dir, h, w).reshape(1, 2)
                                    ])
            final_line = sample_line_evenly(full_pts, N)
        except Exception:
            logger.warning("edge batch: tip-extension failed for frame %d, "
                           "using un-extended backbone", f, exc_info=True)
            final_line = backbone

        try:
            cur_bin  = extract_binary_from_color_image(raw_img, target_color)
            proc_bin = execute_mode_geometry(cur_bin, final_line.tolist(), edge_mode)
            cm = np.zeros((h, w, 3), dtype=np.uint8)
            cm[proc_bin > 0] = [target_color[2], target_color[1], target_color[0]]
            cv2.imwrite(tp, cm)
        except Exception:
            logger.error("edge batch: mask geometry/write failed for frame %d, "
                        "propagated line kept but image not updated for this frame",
                        f, exc_info=True)

        entry = [{"type": "point", "pts": final_line.tolist(), "edge_mode": edge_mode}]
        batch_tl[f]              = entry
        state['timeline'][f]     = entry

    state['prev_line'] = prev_line
    return batch_tl

def propagate_edge_strokes(video_data, frame_range, flat_pts, edge_mode,
                           input_folder, output_folder, target_color,
                           n_samples=25, search_range=12,
                           on_batch_done=None, stop_event=None):
    """Propagate an open-line edge stroke across a frame range in batches.

    Args:
        video_data (h5py.Dataset): Video frames indexed by frame number.
        frame_range (list[int]): Frames to process, starting with the seed frame.
        flat_pts (list): Initial line points as [[x, y], ...].
        edge_mode (str): 'upper_left' or 'lower_right'.
        input_folder (str): Folder containing source mask images.
        output_folder (str): Folder to write propagated mask images to.
        target_color (list[int]): Target BGR color identifying the mask region.
        n_samples (int, optional): Number of points to resample the line to. Defaults to 25.
        search_range (int, optional): Normal-search radius in pixels for refinement. Defaults to 12.
        on_batch_done (callable, optional): Callback invoked with (frame_ids, timeline_entries) after each batch. Defaults to None.
        stop_event (threading.Event, optional): Event used to signal early stop. Defaults to None.

    Returns:
        dict: Timeline mapping frame number to its stroke entry list.
    """
    state   = _edge_setup(video_data, frame_range, flat_pts, edge_mode,
                          input_folder, output_folder, target_color, n_samples, search_range)
    batches = make_batches(frame_range[1:])

    if on_batch_done and state['start_tl']:
        on_batch_done([frame_range[0]], state['start_tl'])

    for batch in batches:
        if stop_event is not None and stop_event.is_set():
            break
        batch_tl = _edge_batch(state, batch, stop_event=stop_event)
        if on_batch_done and batch_tl:
            on_batch_done(list(batch_tl.keys()), batch_tl)

    return state['timeline']


def propagate_edge_masks(h5_path, h5_key, input_folder, output_folder,
                         target_color, start_frame, steps,
                         initial_strokes, default_mode,
                         n_samples=100, search_range=12,
                         on_batch_done=None, stop_event=None):
    """Load video frames from an HDF5 file and propagate an edge stroke across them.

    Args:
        h5_path (str): Path to the HDF5 video file.
        h5_key (str): Dataset key within the HDF5 file.
        input_folder (str): Folder containing source mask images.
        output_folder (str): Folder to write propagated mask images to.
        target_color (list[int]): Target BGR color identifying the mask region.
        start_frame (int): Frame number to start propagation from.
        steps (int): Number of frames to propagate beyond the start frame.
        initial_strokes (list): Initial stroke definitions, each with a 'pts' key.
        default_mode (str): Edge mode to apply, e.g. 'upper_left' or 'lower_right'.
        n_samples (int, optional): Number of points to resample the line to. Defaults to 100.
        search_range (int, optional): Normal-search radius in pixels for refinement. Defaults to 12.
        on_batch_done (callable, optional): Callback invoked with (frame_ids, timeline_entries) after each batch. Defaults to None.
        stop_event (threading.Event, optional): Event used to signal early stop. Defaults to None.

    Returns:
        dict | None: Timeline mapping frame number to its stroke entry list, or None if the file or strokes are missing.
    """
    if not os.path.exists(h5_path) or not initial_strokes:
        return None
    flat_pts = [p for s in initial_strokes for p in s['pts']]
    if not flat_pts:
        return None
    from model.feature_extraction import clear_feature_cache
    clear_propagation_caches()
    clear_feature_cache()
    with h5py.File(h5_path, 'r', locking=False) as f:
        return propagate_edge_strokes(
            f[h5_key], list(range(start_frame, start_frame + steps + 1)),
            flat_pts, default_mode, input_folder, output_folder, target_color,
            n_samples=n_samples, search_range=search_range,
            on_batch_done=on_batch_done, stop_event=stop_event)
