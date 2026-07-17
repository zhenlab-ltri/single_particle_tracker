import os
import copy
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
from utils.poly_utils import (
    sample_contour_evenly,
    optimize_contour_normals,
    smooth_closed_contour,
)


def _poly_setup(video_data, frame_range, flat_pts, edge_mode,
                input_folder, output_folder, target_color, n_samples,
                search_range):
    """Process the start frame of a polygon-stroke propagation run.

    Args:
        video_data (h5py.Dataset): Video frames indexed by frame number.
        frame_range (list[int]): Frames to process, starting with the seed frame.
        flat_pts (list): Initial polygon points as [[x, y], ...].
        edge_mode (str): Polygon geometry mode, e.g. 'hole_fill', 'hole_crop', or 'object'.
        input_folder (str): Folder containing source mask images.
        output_folder (str): Folder to write propagated mask images to.
        target_color (list[int]): Target BGR color identifying the mask region.
        n_samples (int): Number of points to resample the contour to.
        search_range (int): Normal-search radius in pixels for refinement.

    Returns:
        dict: State dict carrying frame data, timeline, and tracking context for later batches.
    """
    h, w         = video_data[frame_range[0]].shape[:2]
    N            = n_samples

    is_same_folder = (input_folder and output_folder and
                      os.path.abspath(input_folder) == os.path.abspath(output_folder))

    flat_pts = sample_contour_evenly(flat_pts, N).astype(int).tolist()

    timeline     = {}
    initial_area = 0
    start_proc   = None

    start_f  = frame_range[0]
    start_tp = find_file_by_frame(output_folder, start_f) or find_file_by_frame(input_folder, start_f)
    if start_tp:
        start_img = cv2.imread(start_tp, cv2.IMREAD_COLOR)
        if start_img is not None:
            start_bin  = extract_binary_from_color_image(start_img, target_color)
            start_proc = execute_mode_geometry(start_bin, flat_pts, edge_mode)
            initial_area = int(np.sum(start_proc > 0))

            cm = np.zeros((h, w, 3), dtype=np.uint8)
            cm[start_proc > 0] = [target_color[2], target_color[1], target_color[0]]
            cv2.imwrite(start_tp, cm)

            timeline[start_f] = [{'type': 'point', 'pts': copy.deepcopy(flat_pts), 'edge_mode': edge_mode}]

    start_tl = {start_f: timeline[start_f]} if start_f in timeline else {}
    return dict(
        video_data=video_data, frame_range=frame_range, edge_mode=edge_mode,
        input_folder=input_folder, output_folder=output_folder, is_same_folder=is_same_folder,
        target_color=target_color, n_samples=N,
        search_range=search_range, h=h, w=w,
        initial_area=initial_area, timeline=timeline, start_tl=start_tl,
        prev_area=initial_area,
    )


def _poly_batch(state, batch, stop_event=None):
    """Track a batch of frames forward and backward from the anchor frame and blend the results.

    Args:
        state (dict): State dict produced by _poly_setup or a prior _poly_batch call.
        batch (list[int]): Frame numbers to process in this batch.
        stop_event (threading.Event, optional): Event used to signal early stop. Defaults to None.

    Returns:
        dict: Timeline entries for the frames processed in this batch, keyed by frame number.
    """
    video_data       = state['video_data']
    frame_range      = state['frame_range']
    edge_mode        = state['edge_mode']
    input_folder     = state['input_folder']
    output_folder    = state['output_folder']
    is_same_folder   = state['is_same_folder']
    target_color     = state['target_color']
    N                = state['n_samples']
    search_range     = state['search_range']
    h, w             = state['h'], state['w']
    initial_area     = state['initial_area']
    prev_area        = state.get('prev_area', initial_area)

    print(f"[poly batch] ENTERED: edge_mode={edge_mode!r} "
          f"batch={list(batch)[:3]}...(n={len(batch)})", flush=True)

    anchor      = frame_range[frame_range.index(batch[0]) - 1]
    batch_range = [anchor] + list(batch)
    total_b     = len(batch_range)

    anchor_pts = state['timeline'][anchor][0]['pts']

    def _bbox(cur_pts):
        """Compute a padded bounding box around a set of contour points.

        Args:
            cur_pts (list): Contour points as [[x, y], ...].

        Returns:
            list[int]: Bounding box [ymin, ymax, xmin, xmax].
        """
        arr = np.array(cur_pts, dtype=np.float32)
        pad = 25
        return [max(0, int(arr[:, 1].min()) - pad), min(h, int(arr[:, 1].max()) + pad),
                max(0, int(arr[:, 0].min()) - pad), min(w, int(arr[:, 0].max()) + pad)]

    local_fwd, local_hist_fwd = {batch_range[0]: anchor_pts}, []
    local_prob_fwd = {}
    true_seed_f      = frame_range[0]
    true_seed_pts    = state['timeline'][true_seed_f][0]['pts']
    true_seed_gray   = to_gray_u8(video_data[true_seed_f])
    true_seed_sample = (true_seed_gray, None, true_seed_pts, _bbox(true_seed_pts))

    g_anchor = to_gray_u8(video_data[anchor])
    safe_track_step(
        g_anchor, g_anchor, None, g_anchor, anchor_pts, h, w,
        resample_fn=lambda p, n: sample_contour_evenly(p, n).astype(int).tolist(),
        n_samples=N, history=local_hist_fwd,
        step_desc="poly warm start fwd", need_prob=True, anchor_sample=true_seed_sample)
    local_hist_fwd.append(true_seed_sample)
    local_hist_fwd.append((g_anchor, None, anchor_pts, _bbox(anchor_pts)))

    for idx in range(total_b - 1):
        fc, fn    = batch_range[idx], batch_range[idx + 1]
        gsrc      = to_gray_u8(video_data[fc])
        gtgt      = to_gray_u8(video_data[fn])
        gsrc_prev = to_gray_u8(video_data[batch_range[idx - 1]]) if idx > 0 else g_anchor
        
        tracked, prob = safe_track_step(
            gsrc, gtgt, gsrc_prev, gsrc, local_fwd[fc], h, w,
            resample_fn=lambda p, n: sample_contour_evenly(p, n).astype(int).tolist(),
            n_samples=N, history=local_hist_fwd,
            step_desc=f"poly local fwd fc={fc}->fn={fn}", need_prob=True,
            anchor_sample=true_seed_sample)
        local_hist_fwd.append((gsrc, gsrc_prev, local_fwd[fc], _bbox(local_fwd[fc])))
        # Index 0 is the permanently-pinned true seed sample -- only prune the rolling
        # recent window after it, never evict the anchor itself.
        if len(local_hist_fwd) > 6:
            local_hist_fwd.pop(1)
        local_fwd[fn] = tracked
        local_prob_fwd[fn] = prob

    local_bwd, local_hist_bwd = {batch_range[-1]: local_fwd[batch_range[-1]]}, []
    local_prob_bwd = {}
    
    anchor_bwd_pts = local_fwd[batch_range[-1]]
    g_anchor_bwd = to_gray_u8(video_data[batch_range[-1]])
    safe_track_step(
        g_anchor_bwd, g_anchor_bwd, None, g_anchor_bwd, anchor_bwd_pts, h, w,
        resample_fn=lambda p, n: sample_contour_evenly(p, n).astype(int).tolist(),
        n_samples=N, history=local_hist_bwd,
        step_desc="poly warm start bwd", need_prob=True, anchor_sample=true_seed_sample)
    local_hist_bwd.append(true_seed_sample)
    local_hist_bwd.append((g_anchor_bwd, None, anchor_bwd_pts, _bbox(anchor_bwd_pts)))

    for idx in range(total_b - 1, 0, -1):
        fc, fp    = batch_range[idx], batch_range[idx - 1]
        gsrc      = to_gray_u8(video_data[fc])
        gtgt      = to_gray_u8(video_data[fp])
        gsrc_prev = to_gray_u8(video_data[batch_range[idx + 1]]) if idx < total_b - 1 else g_anchor_bwd
        
        tracked, prob = safe_track_step(
            gsrc, gtgt, gsrc_prev, gsrc, local_bwd[fc], h, w,
            resample_fn=lambda p, n: sample_contour_evenly(p, n).astype(int).tolist(),
            n_samples=N, history=local_hist_bwd,
            step_desc=f"poly local bwd fc={fc}->fp={fp}", need_prob=True,
            anchor_sample=true_seed_sample)
        local_hist_bwd.append((gsrc, gsrc_prev, local_bwd[fc], _bbox(local_bwd[fc])))
        if len(local_hist_bwd) > 6:
            local_hist_bwd.pop(1)
        local_bwd[fp] = tracked
        local_prob_bwd[fp] = prob

    batch_tl = {}
    prev_pts = anchor_pts
    for f in batch:
        if stop_event is not None and stop_event.is_set():
            break
        tp = (find_file_by_frame(input_folder, f) if is_same_folder
              else find_file_by_frame(output_folder, f) or find_file_by_frame(input_folder, f))
        if not tp:
            print(f"[poly batch] frame {f}: NO FILE FOUND (tp is None) -- skipping frame entirely, "
                  f"input_folder={input_folder!r} output_folder={output_folder!r}", flush=True)
            continue
        raw_img = cv2.imread(tp, cv2.IMREAD_COLOR)
        if raw_img is None:
            print(f"[poly batch] frame {f}: cv2.imread FAILED for tp={tp!r} -- skipping frame entirely",
                  flush=True)
            continue

        gtgt_gray = to_gray_u8(video_data[f])
        inv_gray  = (255 - gtgt_gray).astype(np.float32) / 255.0

        pf_f = local_prob_fwd.get(f)
        pf_b = local_prob_bwd.get(f)
        if pf_f is not None and pf_b is not None:
            pf = np.maximum(pf_f, pf_b)
        elif pf_f is not None:
            pf = pf_f
        elif pf_b is not None:
            pf = pf_b
        else:
            pf = inv_gray

        lf = sample_contour_evenly(local_fwd[f], N).astype(np.float64)
        lb = sample_contour_evenly(local_bwd[f], N).astype(np.float64)
        
        blended      = 0.5 * (lf + lb)
        blended_pts  = blended.astype(int).tolist()

        try:
            print(f"[poly batch] frame {f}: ENTERED LK-tracking + coupling path "
                  f"(mode={edge_mode})", flush=True)
            cur_bin = extract_binary_from_color_image(raw_img, target_color)
            pre_pts = np.array(blended_pts, dtype=np.float64)
            smoothed_pts = smooth_closed_contour(blended_pts, iterations=1, alpha=0.15)
            post_pts     = np.array(smoothed_pts, dtype=np.float64)

            per_point_shift = np.linalg.norm(post_pts - pre_pts, axis=1)
            max_shift        = float(per_point_shift.max()) if len(per_point_shift) else 0.0
            incoherent       = max_shift > 3.0 * search_range
            if incoherent:
                print(f"[poly batch] frame {f}: contour looked incoherent before "
                      f"smoothing (max neighbor-implied shift {max_shift:.1f}px > "
                      f"{3.0*search_range:.0f}px) -- keeping the pre-smoothing edge-search "
                      f"result as-is instead of applying a large corrective smoothing jump",
                      flush=True)
            else:
                blended_pts = [[float(x), float(y)] for x, y in smoothed_pts]

            NUDGE_RANGE = 3
            if pf.max() > 0:
                try:
                    blended_pts = optimize_contour_normals(
                        pf, blended_pts, gtgt_gray, search_range=NUDGE_RANGE)
                except Exception:
                    logger.warning(
                        "poly batch: small edge nudge failed for frame %d, "
                        "keeping LK+smoothing result", f, exc_info=True)

            processed_binary = execute_mode_geometry(cur_bin, blended_pts, edge_mode)
            area = int(np.sum(processed_binary > 0))
            print(f"[poly batch] frame {f}: edge-seeking+coupling area={area} "
                  f"max_point_shift={max_shift:.1f}px", flush=True)

            prev_area      = area
            prev_pts       = blended_pts

            cm = np.zeros((h, w, 3), dtype=np.uint8)
            cm[processed_binary > 0] = [target_color[2], target_color[1], target_color[0]]
            cv2.imwrite(tp, cm)
        except Exception:
            logger.error("poly batch: mask geometry/write failed for frame %d, "
                        "propagated contour kept but image not updated for this frame",
                        f, exc_info=True)

        entry = [{'type': 'point', 'pts': copy.deepcopy(blended_pts), 'edge_mode': edge_mode}]
        batch_tl[f]          = entry
        state['timeline'][f] = entry

    state['prev_area'] = prev_area
    return batch_tl


def propagate_polygon_strokes(video_data, frame_range, flat_pts, edge_mode,
                              input_folder, output_folder, target_color,
                              n_samples=25, search_range=12,
                              on_batch_done=None, stop_event=None):
    """Propagate a closed-polygon stroke across a frame range in batches.

    Args:
        video_data (h5py.Dataset): Video frames indexed by frame number.
        frame_range (list[int]): Frames to process, starting with the seed frame.
        flat_pts (list): Initial polygon points as [[x, y], ...].
        edge_mode (str): Polygon geometry mode, e.g. 'hole_fill', 'hole_crop', or 'object'.
        input_folder (str): Folder containing source mask images.
        output_folder (str): Folder to write propagated mask images to.
        target_color (list[int]): Target BGR color identifying the mask region.
        n_samples (int, optional): Number of points to resample the contour to. Defaults to 25.
        search_range (int, optional): Normal-search radius in pixels for refinement. Defaults to 12.
        on_batch_done (callable, optional): Callback invoked with (frame_ids, timeline_entries) after each batch. Defaults to None.
        stop_event (threading.Event, optional): Event used to signal early stop. Defaults to None.

    Returns:
        dict: Timeline mapping frame number to its polygon entry list.
    """
    state   = _poly_setup(video_data, frame_range, flat_pts, edge_mode,
                          input_folder, output_folder, target_color, n_samples,
                          search_range)
    batches = make_batches(frame_range[1:])

    if on_batch_done and state['start_tl']:
        on_batch_done([frame_range[0]], state['start_tl'])

    for batch in batches:
        if stop_event is not None and stop_event.is_set():
            break
        batch_tl = _poly_batch(state, batch, stop_event=stop_event)
        if on_batch_done and batch_tl:
            on_batch_done(list(batch_tl.keys()), batch_tl)

    return state['timeline']


def propagate_polygon_masks(h5_path, h5_key, input_folder, output_folder,
                            target_color, start_frame, steps,
                            initial_strokes, default_mode,
                            n_samples=75, search_range=12,
                            on_batch_done=None, stop_event=None):
    """Load video frames from an HDF5 file and propagate a polygon stroke across them.

    Args:
        h5_path (str): Path to the HDF5 video file.
        h5_key (str): Dataset key within the HDF5 file.
        input_folder (str): Folder containing source mask images.
        output_folder (str): Folder to write propagated mask images to.
        target_color (list[int]): Target BGR color identifying the mask region.
        start_frame (int): Frame number to start propagation from.
        steps (int): Number of frames to propagate beyond the start frame.
        initial_strokes (list): Initial stroke definitions, each with a 'pts' key.
        default_mode (str): Polygon geometry mode, e.g. 'hole_fill', 'hole_crop', or 'object'.
        n_samples (int, optional): Number of points to resample the contour to. Defaults to 100.
        search_range (int, optional): Normal-search radius in pixels for refinement. Defaults to 12.
        on_batch_done (callable, optional): Callback invoked with (frame_ids, timeline_entries) after each batch. Defaults to None.
        stop_event (threading.Event, optional): Event used to signal early stop. Defaults to None.

    Returns:
        dict | None: Timeline mapping frame number to its polygon entry list, or None if the file or strokes are missing.
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
        return propagate_polygon_strokes(
            f[h5_key], list(range(start_frame, start_frame + steps + 1)),
            flat_pts, default_mode, input_folder, output_folder, target_color,
            n_samples=n_samples, search_range=search_range,
            on_batch_done=on_batch_done, stop_event=stop_event)