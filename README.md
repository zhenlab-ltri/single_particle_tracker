# single_particle_tracker

This pipleine is developed for extracting trajectories of bacteria particles through the pharynx of C. elegans relative to the lumen.

## Running
The code can be installed as is and can be run after setting up a virtual environment with the required modules listed in requirements_simplified.txt.

## Contents

| File | Role |
|---|---|
| `nd2_hdf5.ipynb` | Converts an nd2 file into h5 format with an interactive window for choosing appropriate resolution. |
| `line_generator.py` | Takes in the upper and lower boundaries to generate an approximate centerline of the pharynx. |
| `mask_editor.py` | PyQt5 GUI: multi-layer mask editing, drawing (point/brush/eraser), undo/redo, and the background thread that drives propagation. |
| `propagate_edge.py` | Propagation driver for **open-line** strokes (`upper_left` / `lower_right` edge clipping). |
| `propagate_poly.py` | Propagation driver for **closed-contour** strokes (`hole_fill`, `hole_crop`, `object`). |
| `shared_utils.py` | Common utilities shared by both drivers: grayscale caching, colour-mask (de)coding, Lucas–Kanade point tracking, and mask geometry ops. |
| `edge_utils.py` | Geometry helpers for open-line strokes: even resampling, normal-direction refinement, PCA-based tip extension to the frame boundary. |
| `poly_utils.py` | Geometry helpers for closed-contour strokes: even resampling, normal-direction refinement, Laplacian smoothing. |
| `regressor.py` | `ConformalDeepForestRegressor` — trains a fresh CDForest classifier per frame/ROI and produces a per-pixel foreground-probability field. |
| `cascade_layer.py` | `CDForestCascadeLayer` — one cascade layer of four forests with CDUM-based uncertainty filtering, weighting, and conformal aggregation. |
| `feature_extraction.py` | Multiscale per-pixel feature extraction. |
| `trajectory.ipynb` | Generate centerpoint of the lumen and visualize trajectory. |
| `plotter.ipynb` | Find the trapping points (maxima) and plot the data for all instances. |


## Pipeline overview

```
nd2 is converted to h5 at an appropriate resolution
            │
            ▼
 User draws stroke on frame N
            │
            ▼
 mask_editor.py ── trigger_external_propagation()
            │
            ▼
 propagate_edge.py / propagate_poly.py 
            │
    ┌───────┴────────┐
    ▼                ▼
 Lucas–Kanade      ConformalDeepForestRegressor
 point tracking    (regressor.py)
 (shared_utils.py)        │
    │              ┌──────┴───────┐
    │              ▼              ▼
    │        feature_extraction.py  cascade_layer.py
    │        (multiscale pixel      (CDForest cascade,
    │         features)              CDUM uncertainty)
    │                     │
    └─────────┬───────────┘
              ▼
   per-pixel probability field
              │
              ▼
   edge_utils.py / poly_utils.py
   (normal-direction search, smoothing,
    area/shape constraints)
              │
              ▼
   updated binary mask written to disk
   (entire process used to generate upper boundary, lower boundary, and fixed point)
              │
              ▼
   line_generator.py is used to generate a rough centerline
              │
              ▼
    trajectory.ipynb is used to write a csv that tracks the positions of
    bacteria particle relative to the centre of the pharynx lumen
              │
              ▼
     trajectories are plotted
```

## Mask generation
First, the obtained nd2 recordings are covnerted to h5 format. Then, a PyQt5 mask-annotation editor with automated mask propagation across video frames is used to generate a fixed point (a cell on the pharynx wall usually), upper, and lower boundaries of the pharynx. The user draws a guideline on a single frame, and the pipeline propagates that correction forward/backward across a frame range by re-training a lightweight per-object classifier at every step and searching for the true boundary in its vicinity.

For each target frame, the drivers combine two independent estimates of where the boundary should be:

1. **Motion estimate** — forward/backward Lucas–Kanade optical flow of the stroke's sample points, run from the last confirmed frame in both directions and blended (`shared_utils.track_one_step_cdum`).
2. **Appearance estimate** — a `ConformalDeepForestRegressor` is retrained per step directly on the current and recent-history frames (boundary-band foreground vs. deep-interior/exterior background for polygons, above/below-line brightness comparison for open edges), producing a per-pixel probability field.

The blended point estimate is then refined by searching along the outward normal of each boundary point for the peak of the probability field (`optimize_line_normals` / `optimize_contour_normals`), followed by mode-specific post-processing.

The per-frame probability field is produced by a simplified from-scratch reimplementation of **CDForest**, an uncertainty-aware deep-forest architecture, adapted here for binary pixel foreground/background classification instead of its original tabular/image classification setting:

> Zhang, J., Qiu, Y., & Dong, L. (2025). *Conformal deep forest for uncertainty-aware classification*. Journal of King Saud University – Computer and Information Sciences, 37(6), article 155. https://doi.org/10.1007/s44443-025-00175-3 — https://link.springer.com/article/10.1007/s44443-025-00175-3

CDForest extends the deep forest / gcForest cascade with three modules built on the paper's **conformal dual uncertainty metric (CDUM)**, which scores each class prediction of each forest from two directions: how much *other* classes' predictions intrude on it (passive uncertainty) and how much it dominates *other* classes' predictions (active uncertainty). In this codebase:

- **`cascade_layer.CDForestCascadeLayer`** — one cascade layer of four forests (two `RandomForestClassifier`, two `ExtraTreesClassifier`). It computes class-conditional inductive conformal p-values (CICP) per forest, derives CDUM values from them, and uses those values for:
  - **Layer-wise uncertainty filtering** (`transform_and_filter`) — zeroing out a forest's output for a class when its normalized CDUM exceeds the layer's quantile threshold, before that output is passed to the next cascade layer.
  - **Class-wise inference weighting** (`inference_weights`) — down-weighting forests that are more uncertain about a given class when aggregating predictions.
  - **Conformal p-value aggregation** (`conformal_pvalues`) — producing the per-forest p-values consumed by the weighted aggregation at the output layer.
- **`regressor.ConformalDeepForestRegressor`** — orchestrates the cascade (up to `MAX_LAYERS = 5`, stopped early via validation AUC), builds the pixel-level training signal, and converts the aggregated output-layer p-values into two per-pixel fields: a raw appearance field (the genuine local foreground signal, used for boundary/normal search) and a spatially-modulated field (appearance × a Gaussian centered on the object's centroid, used to suppress far-away lookalikes).

Because a fresh classifier is trained per frame/ROI from a handful of labeled bands (boundary-band vs. interior/exterior for polygons, above/below-line for open edges), CDForest's data-efficient, hyperparameter-light, and interpretable cascade design is a good fit compared to a large pretrained segmentation network: training happens online, per object, per frame, with very few labeled pixels and a tight latency budget.

These annotation modes are available to allow for easier mask annotation/correction while propagating.

| Mode | Stroke type | Geometry op |
|---|---|---|
| `upper_left` / `lower_right` | open line | Clip the mask above/below the drawn line, refined via `edge_utils` normal search + PCA tip extension. |
| `hole_fill` | closed polygon | Fill the enclosed region into the mask. |
| `hole_crop` | closed polygon | Remove the enclosed region from the mask. |
| `object` | closed polygon | Replace the mask with the polygon itself. |

Frames are processed in randomly-sized batches of 1–5 so the GUI can show incremental progress; each batch tracks forward from an anchor frame and backward from the batch's far end, then blends the two directions.

The propagation entry point is in `mask_editor.py`, which expects a background video source (HDF5 dataset, image folder, or TIFF) and one or more mask layers (folders of colour-coded PNG/TIFF masks, one colour per layer). Draw a guideline with the Point or Brush tool, pick an edge/clipping mode, and use **Propagate** to run the pipeline described above in a background thread. Multiple layers can be queued for propagation at the same time but the queue only works in batch-level. Hole-fill and hole-crop are only used for making corrections easier.

## Centerline

The upper and lower boundaries are then used to create a rough centerline of the pharynx. This was done by extracting the boundary points, then averaging and resampling them to fit a parametric curve to make it run acorss the frame. A baseline was constructed and a simplified discrete fourier transform was performed via matrix multiplication to add some noise that allows for the capture of small movements of the pharynx wall. If any major bends appear in the footage, the baseline can be easily made a polyfit (or whatever matches the required shape) but for most of the footage that was used for this project, the pharynx within the frame was straight enough with the exception of a few. A small temporal and spatial refinement was added to preserve the overall shape. A csv with the centerline coordinates spaced evenly and overlay video will be saved to the chosen output path.

## Center of the Lumen & Trajectory

Along with upper and lower boudnaries, masks of a fixed point were generated as well. The centerline and this fixed point were used together to obtain a rough location of of lumen center.  First, a frame is chosen by the user for callibration. The user clicks the location of the point to track. The fixed point's projection onto the centerline is used to create a transformed plane: direction of projection and perpendicualar. The location of the center is calculated in this new plane. In the remainig frames, the same offsets are applied from the projection of the fixed point onto the centerline. The node closest to that offset on the centerline is treated to be the center. The window then displays the chosen point from start frame to end frame. If the user is not satisfied, we can choose once again (same or different frame) to callibrate. Once satisfied, the user can save the csv which has frame number, longitudinal distance from center point, transverse deviation from centerline, x and y coordinates of center point in that order. This csv can be used to visualize the trajectory of the particle. 

## Plotting


## Caveats 
1. This training model predicts accurately only on noisy frames where the transitions between light/dark intensities or structures are apparent. High resolution images may result in the generation of conflicting probability fields by the model. 
2. The predicitons are only an estimate and drifts can accumulate over a larger amount of frames. Therefore, it is recommended that users run it for 1-60 frames at a time and redraw the guideline for the following ones.
3. The features are extracted on inverted pixels since the purpose of this model is to recognize darker features the region of interest. User have to change this in model/feature_extraction.py if they wish to track lighter structures.
4. In order to load the background frames and layer frames from a folder of images, it was asusmed that the name of the files is a frame number so that the gui can easily match different layers with the background image. If the user's files have different naming conventions, this should be changed accordingly in mask_editor.py
5. Only one line can be tracked at a time within a single layer with this mask_generation gui due to the format in which the drawn lines are processed. 
6. Both the centerline and the center of the lumen are not absolute and can have small deviations.
