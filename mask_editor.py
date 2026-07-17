import os
import re
import sys
import copy
import logging
import threading
import traceback
import warnings
import contextlib
import cv2
import h5py
import numpy as np

logger = logging.getLogger(__name__)

from PyQt5.QtCore import Qt, QPointF, QPoint, QThread, QObject, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QComboBox, QFileDialog, QMessageBox,
    QFrame, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QSlider, QSpinBox,
    QLineEdit, QButtonGroup, QToolBar, QAction, QSizePolicy, QActionGroup,
    QScrollArea, QColorDialog, QSplitter, QStackedWidget, QDoubleSpinBox
)
from PyQt5.QtGui import QImage, QPixmap, QPen, QColor, QBrush, QIcon, QCursor, QPainter


LAYER_DEFAULT_COLORS = [
    QColor(0,   200,  80,  255),
    QColor(30,  120, 255,  255),
    QColor(255, 180,   0,  255),
    QColor(220,  60, 220,  255),
    QColor(0,   210, 230,  255),
    QColor(255,  90,  40,  255),
]

EDGE_MODE_KEYS = {
    Qt.Key_1: "upper_left",
    Qt.Key_2: "lower_right",
    Qt.Key_3: "hole_fill",
    Qt.Key_4: "hole_crop",
    Qt.Key_5: "object",
}
EDGE_MODE_COMBO_INDEX = {
    "upper_left":  0,
    "lower_right": 1,
    "hole_fill":   2,
    "hole_crop":   3,
    "object":      4,
}


def extract_boundary_points(binary_mask, boundary_type):
    """Extract the upper or lower boundary points of a binary mask, per column.

    Cleans the mask with a morphological close then open, then finds, for each
    column containing foreground pixels, the first foreground pixel from the
    requested side.

    Args:
        binary_mask (np.ndarray): Single-channel binary mask (foreground > 0).
        boundary_type (str): Which boundary to extract, 'upper' or 'lower'.

    Returns:
        tuple: (points, raw_min_x, raw_max_x) where points is an (N, 2) array of
            (x, y) boundary coordinates sorted by x, and raw_min_x/raw_max_x are
            the minimum and maximum x with a valid point (0 if none found).
    """
    h, w = binary_mask.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
    has_pixel = np.any(cleaned_mask > 0, axis=0)
    valid_x = np.where(has_pixel)[0]
    if len(valid_x) == 0:
        return np.empty((0, 2)), 0, 0
    if boundary_type == 'upper':
        flipped_mask = cleaned_mask[::-1, :]
        first_y_from_bottom = np.argmax(flipped_mask > 0, axis=0)
        valid_y = (h - 1) - first_y_from_bottom[valid_x]
    else:
        valid_y = np.argmax(cleaned_mask > 0, axis=0)[valid_x]
    points = np.column_stack((valid_x, valid_y.astype(np.float64)))
    sort_idx = np.argsort(points[:, 0])
    points = points[sort_idx]
    raw_min_x = points[0, 0]
    raw_max_x = points[-1, 0]
    return points, raw_min_x, raw_max_x



class MaskLayer:
    def __init__(self, input_folder, output_folder, color):
        """Initialize a mask layer with its source/output folders, color, and empty history.

        Args:
            input_folder (str): Directory containing the layer's source mask images.
            output_folder (str): Directory where edited mask images are written.
            color (QColor): Display and encoding color used to render this layer's mask.
        """
        self.input_folder  = input_folder
        self.output_folder = output_folder
        self.color         = color
        self.mask_files    = []
        self.current_mask_raw = None
        self.strokes       = []
        self.last_drawn_strokes_buffer = []
        self.undo_stacks   = {}
        self.redo_stacks   = {}
        self.visible       = True
        self.frame_guidelines = {}
        self.prop_settings = {
            'samples':      60,
            'search_range': 12,
        }

    def display_name(self, idx):
        """Build a display label for this layer from its position and input folder name.

        Args:
            idx (int): Zero-based index of this layer in the layer list.

        Returns:
            str: Human-readable label such as "Layer 1: folder_name".
        """
        folder = os.path.basename(self.input_folder) or self.input_folder
        return f"Layer {idx+1}: {folder}"


def layer_propagation_mode(layer, default='upper_left'):
    """Determine the edge/clipping mode to use when propagating a layer.

    Args:
        layer (MaskLayer): Layer whose most recent stroke's edge mode is inspected.
        default (str): Mode returned when the layer has no strokes with an edge mode.

    Returns:
        str: The edge mode name, e.g. 'upper_left', 'lower_right', 'hole_fill'.
    """
    for stroke in reversed(layer.strokes):
        m = stroke.get('edge_mode')
        if m:
            return m
    return default



class ViewportCanvas(QGraphicsView):
    def __init__(self, parent_app):
        """Set up the graphics view, scene, and drawing state for the mask canvas.

        Args:
            parent_app (PyQtMaskEditorWorkspace): Main window that owns this canvas.
        """
        super().__init__()
        self.app = parent_app
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setBackgroundBrush(QBrush(QColor("#1e1e1e")))

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setRenderHint(QPainter.SmoothPixmapTransform)

        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        self.node_items  = []
        self.line_items  = []

        self.dragged_node_idx        = None
        self.node_radius             = 5
        self._brush_active           = False
        self._current_brush_stroke_idx = None

    def resizeEvent(self, event):
        """Refit the displayed image into the view whenever the widget is resized.

        Args:
            event (QResizeEvent): Resize event triggering this handler.
        """
        super().resizeEvent(event)
        if self.pixmap_item.pixmap() and not self.pixmap_item.pixmap().isNull():
            zoom_val = getattr(self.app, 'zoom_slider', None)
            if zoom_val is not None and self.app.zoom_slider.value() != 100:
                self.app._update_zoom(self.app.zoom_slider.value())
            else:
                self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def _active_layer(self):
        """Get the currently active mask layer from the parent application.

        Returns:
            MaskLayer | None: The active layer, or None if there isn't one.
        """
        return self.app.active_layer()

    def _in_image(self, x, y):
        """Check whether an image-space coordinate falls inside the active layer's mask.

        Args:
            x (int): Column coordinate in image space.
            y (int): Row coordinate in image space.

        Returns:
            bool: True if the point lies within the active layer's mask bounds.
        """
        layer = self._active_layer()
        if layer is None or layer.current_mask_raw is None:
            return False
        h, w = layer.current_mask_raw.shape
        return 0 <= x < w and 0 <= y < h

    def _clamp(self, x, y):
        """Clamp an image-space coordinate to the active layer's mask bounds.

        Args:
            x (int): Column coordinate in image space.
            y (int): Row coordinate in image space.

        Returns:
            tuple: (x, y) clamped to the valid range of the active layer's mask.
        """
        layer = self._active_layer()
        h, w = layer.current_mask_raw.shape
        return max(0, min(w - 1, x)), max(0, min(h - 1, y))

    def _scene_to_img(self, scene_pos):
        """Convert a scene coordinate to integer image-space pixel coordinates.

        Args:
            scene_pos (QPointF): Position in scene coordinates.

        Returns:
            tuple: (x, y) integer pixel coordinates.
        """
        return int(scene_pos.x()), int(scene_pos.y())

    def get_node_at_position(self, scene_pos):
        """Find the point-stroke node nearest a scene position, within the hit radius.

        Args:
            scene_pos (QPointF): Position in scene coordinates to test.

        Returns:
            tuple | None: (stroke_index, point_index) of the hit node, or None if no
                node is close enough.
        """
        layer = self._active_layer()
        if layer is None:
            return None
        r2 = self.node_radius * self.node_radius * 2
        for si, stroke in enumerate(layer.strokes):
            if stroke['type'] != 'point':
                continue
            for pi, pt in enumerate(stroke['pts']):
                dx = scene_pos.x() - pt[0]
                dy = scene_pos.y() - pt[1]
                if dx*dx + dy*dy <= r2:
                    return (si, pi)
        return None

    def erase_at(self, scene_pos):
        """Remove stroke points that fall within the eraser radius of a scene position.

        Args:
            scene_pos (QPointF): Center of the eraser in scene coordinates.
        """
        layer = self._active_layer()
        if layer is None:
            return
        r = self.app.eraser_size
        r2 = r * r
        new_strokes = []
        changed = False
        for stroke in layer.strokes:
            new_pts = [pt for pt in stroke['pts']
                       if (scene_pos.x()-pt[0])**2 + (scene_pos.y()-pt[1])**2 > r2]
            if len(new_pts) != len(stroke['pts']):
                changed = True
            if new_pts:
                new_strokes.append({'type': stroke['type'], 'pts': new_pts,
                                    'edge_mode': stroke.get('edge_mode', self.app.edge_mode)})
        if len(new_strokes) != len(layer.strokes):
            changed = True
        layer.strokes = new_strokes
        if changed:
            self.redraw_guidelines()

    def mousePressEvent(self, event):
        """Handle a mouse press by starting a brush stroke, adding/dragging a point, or erasing.

        Args:
            event (QMouseEvent): Mouse press event to handle.
        """
        layer = self._active_layer()
        if layer is None or layer.current_mask_raw is None:
            super().mousePressEvent(event)
            return

        scene_pos = self.mapToScene(event.pos())
        img_x, img_y = self._scene_to_img(scene_pos)
        mode = self.app.draw_mode
        current_edge_mode = self.app.edge_mode

        if mode == 'eraser':
            if event.button() == Qt.LeftButton:
                self.erase_at(scene_pos)
                event.accept()
            return

        if mode == 'brush':
            if event.button() == Qt.LeftButton:
                if self._in_image(img_x, img_y):
                    self._brush_active = True
                    new_stroke = {'type': 'brush', 'pts': [(img_x, img_y)],
                                  'edge_mode': current_edge_mode}
                    layer.strokes.append(new_stroke)
                    self._current_brush_stroke_idx = len(layer.strokes) - 1
                    self.redraw_guidelines()
                    event.accept()
            return

        if mode == 'point':
            node = self.get_node_at_position(scene_pos)
            if node is not None:
                si, pi = node
                if event.button() == Qt.LeftButton:
                    self.dragged_node_idx = (si, pi)
                    self.setCursor(Qt.ClosedHandCursor)
                    event.accept()
                    return
                elif event.button() == Qt.RightButton:
                    layer.strokes[si]['pts'].pop(pi)
                    if not layer.strokes[si]['pts']:
                        layer.strokes.pop(si)
                    self.redraw_guidelines()
                    event.accept()
                    return
            else:
                if event.button() == Qt.LeftButton and self._in_image(img_x, img_y):
                    if layer.strokes and layer.strokes[-1]['type'] == 'point':
                        layer.strokes[-1]['pts'].append((img_x, img_y))
                    else:
                        layer.strokes.append({'type': 'point', 'pts': [(img_x, img_y)],
                                              'edge_mode': current_edge_mode})
                    self.redraw_guidelines()
                    event.accept()
                    return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse movement by continuing an active brush stroke, drag, or erase.

        Args:
            event (QMouseEvent): Mouse move event to handle.
        """
        layer = self._active_layer()
        scene_pos = self.mapToScene(event.pos())
        mode = self.app.draw_mode

        if mode == 'eraser' and event.buttons() & Qt.LeftButton:
            self.erase_at(scene_pos)
            event.accept()
            return

        if mode == 'brush' and self._brush_active and self._current_brush_stroke_idx is not None and layer is not None:
            img_x, img_y = self._scene_to_img(scene_pos)
            if self._in_image(img_x, img_y):
                layer.strokes[self._current_brush_stroke_idx]['pts'].append((img_x, img_y))
                self.redraw_guidelines()
            event.accept()
            return

        if mode == 'point' and self.dragged_node_idx is not None and layer is not None and layer.current_mask_raw is not None:
            si, pi = self.dragged_node_idx
            img_x, img_y = self._scene_to_img(scene_pos)
            img_x, img_y = self._clamp(img_x, img_y)
            layer.strokes[si]['pts'][pi] = (img_x, img_y)
            self.redraw_guidelines()
            event.accept()
            return

        if mode == 'point':
            if self.get_node_at_position(scene_pos) is not None:
                self.setCursor(Qt.PointingHandCursor)
            else:
                self.setCursor(Qt.ArrowCursor)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Finish the active brush stroke or node drag and save it to the undo history.

        Args:
            event (QMouseEvent): Mouse release event to handle.
        """
        layer = self._active_layer()
        if self._brush_active:
            self._brush_active = False
            self._current_brush_stroke_idx = None
            if layer:
                self.app.save_stroke_history_for_frame(layer)
            event.accept()
            return
        if self.dragged_node_idx is not None:
            self.dragged_node_idx = None
            self.setCursor(Qt.ArrowCursor)
            if layer:
                self.app.save_stroke_history_for_frame(layer)
            event.accept()
            return
        if event.button() == Qt.LeftButton and self.app.draw_mode == 'point' and layer:
            self.app.save_stroke_history_for_frame(layer)
        super().mouseReleaseEvent(event)

    def redraw_guidelines(self):
        """Redraw every visible layer's stroke nodes and lines in the scene."""
        for item in self.node_items:
            self.scene.removeItem(item)
        for item in self.line_items:
            self.scene.removeItem(item)
        self.node_items.clear()
        self.line_items.clear()

        if not self.app.layers:
            return

        pen_node = QPen(QColor("white"), 1, Qt.SolidLine)

        for idx, layer in enumerate(self.app.layers):
            if not layer.visible or not layer.strokes:
                continue

            if idx == self.app.active_layer_idx:
                pen_point_line = QPen(QColor("#000000"), 2, Qt.SolidLine)
                pen_brush_line = QPen(QColor("#cc0000"), 1.5, Qt.SolidLine)
            else:
                pen_point_line = QPen(QColor("#444444"), 1, Qt.DashLine)
                pen_brush_line = QPen(QColor("#aa5555"), 1, Qt.DashLine)

            pen_brush_line.setCapStyle(Qt.RoundCap)
            pen_brush_line.setJoinStyle(Qt.RoundJoin)

            for stroke in layer.strokes:
                pts = stroke['pts']
                if not pts:
                    continue
                stroke_edge_mode = stroke.get('edge_mode', self.app.edge_mode)
                hole_mode = stroke_edge_mode in ("hole_fill", "hole_crop")
                is_brush = stroke['type'] == 'brush'
                if is_brush:
                    for i in range(1, len(pts)):
                        prev, cur = pts[i-1], pts[i]
                        line = self.scene.addLine(prev[0], prev[1], cur[0], cur[1], pen_brush_line)
                        self.line_items.append(line)
                else:
                    node_color = layer.color
                    brush_fill = QBrush(node_color)
                    for i, pt in enumerate(pts):
                        r = self.node_radius if idx == self.app.active_layer_idx else self.node_radius - 1
                        node = self.scene.addEllipse(pt[0]-r, pt[1]-r, r*2, r*2, pen_node, brush_fill)
                        self.node_items.append(node)
                        if i > 0:
                            prev = pts[i-1]
                            line = self.scene.addLine(prev[0], prev[1], pt[0], pt[1], pen_point_line)
                            self.line_items.append(line)
                    if hole_mode and len(pts) > 2:
                        pen_close = QPen(QColor("#000000"), 1, Qt.DashLine)
                        line = self.scene.addLine(pts[-1][0], pts[-1][1], pts[0][0], pts[0][1], pen_close)
                        self.line_items.append(line)



class LayerRowWidget(QFrame):
    def __init__(self, layer_idx, layer, parent_app):
        """Store references and build the row widget for a single mask layer.

        Args:
            layer_idx (int): Index of this layer within the workspace's layer list.
            layer (MaskLayer): Layer data model this row displays and edits.
            parent_app (PyQtMaskEditorWorkspace): Main window that owns this row.
        """
        super().__init__()
        self.layer_idx  = layer_idx
        self.layer      = layer
        self.parent_app = parent_app
        self.is_expanded = False
        self._build()

    def _build(self):
        """Construct the row's header controls and collapsible settings panel."""
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Plain)

        bg_color = "#eaf2ff" if self.layer_idx == self.parent_app.active_layer_idx else "#fafafa"
        border_color = "#4a90e2" if self.layer_idx == self.parent_app.active_layer_idx else "#dcdcdc"

        self.setStyleSheet(
            f"QFrame {{ border: 1px solid {border_color}; border-radius: 4px; margin: 1px 0; background-color: {bg_color}; }}"
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(0)

        header = QHBoxLayout()
        header.setContentsMargins(2, 2, 2, 2)
        header.setSpacing(6)

        self.btn_toggle_expand = QPushButton("▶")
        self.btn_toggle_expand.setFixedSize(20, 20)
        self.btn_toggle_expand.setStyleSheet("QPushButton { border: none; font-size: 8pt; font-weight: bold; color: #555; background: transparent; }")
        self.btn_toggle_expand.clicked.connect(self._toggle_expand_panel)
        header.addWidget(self.btn_toggle_expand)

        self.btn_color = QPushButton()
        self.btn_color.setFixedSize(16, 16)
        self.btn_color.setToolTip("Change layer colour")
        self._apply_color_swatch()
        self.btn_color.clicked.connect(self._pick_color)
        header.addWidget(self.btn_color)

        self.lbl_name = QLabel(f"Layer {self.layer_idx + 1}")
        self.lbl_name.setStyleSheet("font-weight: bold; font-size: 9pt; border: none; color: #333; background: transparent;")
        header.addWidget(self.lbl_name)
        header.addStretch()

        self.btn_vis = QPushButton("👁")
        self.btn_vis.setFixedSize(24, 24)
        self.btn_vis.setCheckable(True)
        self.btn_vis.setChecked(True)
        self.btn_vis.setStyleSheet(
            "QPushButton { border: none; font-size: 11pt; color: #333; background: transparent; }"
            "QPushButton:!checked { color: #ccc; }"
        )
        self.btn_vis.toggled.connect(self._toggle_visibility)
        header.addWidget(self.btn_vis)

        btn_remove = QPushButton("✕")
        btn_remove.setFixedSize(22, 22)
        btn_remove.setStyleSheet("QPushButton { border: none; color: #ba2525; font-weight: bold; font-size: 9pt; background: transparent; }")
        btn_remove.clicked.connect(lambda: self.parent_app.remove_layer(self.layer_idx))
        header.addWidget(btn_remove)

        outer.addLayout(header)

        self.paths_panel = QWidget()
        self.paths_panel.setVisible(False)
        panel_layout = QVBoxLayout(self.paths_panel)
        panel_layout.setContentsMargins(22, 4, 4, 4)
        panel_layout.setSpacing(4)

        in_row = QHBoxLayout()
        in_row.setSpacing(4)
        lbl_in = QLabel("In:")
        lbl_in.setFixedWidth(24)
        in_row.addWidget(lbl_in)
        self.txt_in = QLineEdit(self.layer.input_folder)
        self.txt_in.setStyleSheet("font-size: 8pt; padding: 2px; background: white;")
        self.txt_in.editingFinished.connect(self._input_edited)
        in_row.addWidget(self.txt_in)
        btn_in = QPushButton("…")
        btn_in.setFixedSize(22, 18)
        btn_in.clicked.connect(self._browse_input)
        in_row.addWidget(btn_in)
        panel_layout.addLayout(in_row)

        out_row = QHBoxLayout()
        out_row.setSpacing(4)
        lbl_out = QLabel("Out:")
        lbl_out.setFixedWidth(24)
        out_row.addWidget(lbl_out)
        self.txt_out = QLineEdit(self.layer.output_folder)
        self.txt_out.setStyleSheet("font-size: 8pt; padding: 2px; background: white;")
        self.txt_out.editingFinished.connect(self._output_edited)
        out_row.addWidget(self.txt_out)
        btn_out = QPushButton("…")
        btn_out.setFixedSize(22, 18)
        btn_out.clicked.connect(self._browse_output)
        out_row.addWidget(btn_out)
        panel_layout.addLayout(out_row)

        prop_sep = QFrame()
        prop_sep.setFrameShape(QFrame.HLine)
        prop_sep.setStyleSheet("color: #ddd;")
        panel_layout.addWidget(prop_sep)

        prop_hdr = QLabel("Propagation settings")
        prop_hdr.setStyleSheet("font-size: 8pt; font-weight: bold; color: #555; border: none;")
        panel_layout.addWidget(prop_hdr)

        samples_row = QHBoxLayout()
        samples_row.addWidget(QLabel("Samples:"))
        self.spin_layer_samples = QSpinBox()
        self.spin_layer_samples.setRange(10, 500)
        self.spin_layer_samples.setValue(self.layer.prop_settings['samples'])
        self.spin_layer_samples.setFixedWidth(60)
        self.spin_layer_samples.valueChanged.connect(
            lambda v: self.layer.prop_settings.__setitem__('samples', v))
        samples_row.addWidget(self.spin_layer_samples)
        samples_row.addStretch()
        panel_layout.addLayout(samples_row)

        self.prop_param_stack = QStackedWidget()

        edge_pg        = QWidget()
        edge_pg_layout = QHBoxLayout(edge_pg)
        edge_pg_layout.setContentsMargins(0, 0, 0, 0)
        edge_pg_layout.addWidget(QLabel("Search:"))
        self.spin_layer_search = QSpinBox()
        self.spin_layer_search.setRange(1, 60)
        self.spin_layer_search.setValue(self.layer.prop_settings['search_range'])
        self.spin_layer_search.setFixedWidth(55)
        self.spin_layer_search.valueChanged.connect(
            lambda v: self.layer.prop_settings.__setitem__('search_range', v))
        edge_pg_layout.addWidget(self.spin_layer_search)
        edge_pg_layout.addStretch()
        self.prop_param_stack.addWidget(edge_pg)

        poly_pg        = QWidget()
        poly_pg_layout = QVBoxLayout(poly_pg)
        poly_pg_layout.setContentsMargins(0, 0, 0, 0)
        poly_pg_layout.addStretch()

        self.prop_param_stack.addWidget(poly_pg)

        panel_layout.addWidget(self.prop_param_stack)
        self._sync_prop_param_stack()

        outer.addWidget(self.paths_panel)

    def _sync_prop_param_stack(self):
        """Show the edge or polygon propagation parameter page matching the layer's mode."""
        mode    = layer_propagation_mode(self.layer, default=self.parent_app.edge_mode)
        is_poly = mode in ('hole_fill', 'hole_crop', 'object')
        self.prop_param_stack.setCurrentIndex(1 if is_poly else 0)

    def mousePressEvent(self, event):
        """Make this layer the active layer when its row is left-clicked.

        Args:
            event (QMouseEvent): Mouse press event to handle.
        """
        if event.button() == Qt.LeftButton:
            self.parent_app.set_active_layer_by_index(self.layer_idx)
            event.accept()
        else:
            super().mousePressEvent(event)

    def _toggle_expand_panel(self):
        """Show or hide the row's input/output/propagation settings panel."""
        self.is_expanded = not self.is_expanded
        self.btn_toggle_expand.setText("▼" if self.is_expanded else "▶")
        self.paths_panel.setVisible(self.is_expanded)
        if self.is_expanded:
            self._sync_prop_param_stack()

    def _apply_color_swatch(self):
        """Update the color swatch button's stylesheet to match the layer's color."""
        c = self.layer.color
        self.btn_color.setStyleSheet(
            f"QPushButton {{ background: rgba({c.red()},{c.green()},{c.blue()},{c.alpha()}); "
            f"border: 1px solid #666; border-radius: 2px; }}"
        )

    def _pick_color(self):
        """Open a color picker and apply the chosen color to the layer."""
        col = QColorDialog.getColor(self.layer.color, self, "Choose layer colour")
        if col.isValid():
            self.layer.color = col
            self._apply_color_swatch()
            self.parent_app.render_current_workspace_view()

    def _toggle_visibility(self, checked):
        """Toggle the layer's visibility and re-render the workspace view.

        Args:
            checked (bool): New visibility state of the layer.
        """
        self.layer.visible = checked
        self.parent_app.render_current_workspace_view()

    def _browse_input(self):
        """Prompt for a new input folder and reload the layer's masks from it."""
        folder = QFileDialog.getExistingDirectory(self, "Select mask input folder")
        if folder:
            self.layer.input_folder = folder
            self.txt_in.setText(folder)
            self.parent_app.reload_layer(self.layer_idx)

    def _browse_output(self):
        """Prompt for a new output folder and reload the layer."""
        folder = QFileDialog.getExistingDirectory(self, "Select output folder")
        if folder:
            self.layer.output_folder = folder
            self.txt_out.setText(folder)
            self.parent_app.reload_layer(self.layer_idx)

    def _input_edited(self):
        """Apply a manually typed input folder path if it is a valid directory."""
        path = self.txt_in.text().strip()
        if os.path.isdir(path):
            self.layer.input_folder = path
            self.parent_app.reload_layer(self.layer_idx)

    def _output_edited(self):
        """Apply a manually typed output folder path."""
        path = self.txt_out.text().strip()
        if path:
            self.layer.output_folder = path
            self.parent_app.reload_layer(self.layer_idx)



class CreateLayerDialog(QWidget):
    """Dialog to create a new blank-mask layer at a chosen path and frame range."""

    def __init__(self, parent=None, frame_size=None):
        """Configure the dialog window and store the background frame size for blank masks.

        Args:
            parent (QWidget | None): Parent widget for this dialog.
            frame_size (tuple | None): (height, width) used to size new blank mask
                images, or None if no background is loaded.
        """
        super().__init__(parent, Qt.Dialog | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        self.setWindowTitle("Create New Layer")
        self.setFixedSize(440, 260)
        self.result_folder = None
        self._frame_size = frame_size
        self._name_edited_by_user = False
        self._build()

    def _build(self):
        """Build the dialog's path, frame range, info, and action button widgets."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        path_lbl = QLabel("Save folder:")
        path_lbl.setStyleSheet("font-size: 9pt; font-weight: bold;")
        layout.addWidget(path_lbl)

        path_row = QHBoxLayout()
        self.txt_path = QLineEdit()
        self.txt_path.setPlaceholderText("Choose a parent directory…")
        path_row.addWidget(self.txt_path)
        btn_browse = QPushButton("Browse")
        btn_browse.setFixedWidth(64)
        btn_browse.clicked.connect(self._browse)
        path_row.addWidget(btn_browse)
        layout.addLayout(path_row)

        name_lbl = QLabel("Folder name:")
        name_lbl.setStyleSheet("font-size: 9pt; font-weight: bold;")
        layout.addWidget(name_lbl)

        self.txt_folder_name = QLineEdit()
        self.txt_folder_name.setPlaceholderText("frame 0 - frame 100")
        self.txt_folder_name.textEdited.connect(self._on_folder_name_edited)
        layout.addWidget(self.txt_folder_name)

        range_lbl = QLabel("Frame range:")
        range_lbl.setStyleSheet("font-size: 9pt; font-weight: bold;")
        layout.addWidget(range_lbl)

        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("Start index:"))
        self.spin_start = QSpinBox()
        self.spin_start.setMinimum(0)
        self.spin_start.setMaximum(999999)
        self.spin_start.setValue(0)
        self.spin_start.setFixedWidth(80)
        range_row.addWidget(self.spin_start)
        range_row.addSpacing(16)
        range_row.addWidget(QLabel("End index:"))
        self.spin_end = QSpinBox()
        self.spin_end.setMinimum(0)
        self.spin_end.setMaximum(999999)
        self.spin_end.setValue(100)
        self.spin_end.setFixedWidth(80)
        range_row.addWidget(self.spin_end)
        range_row.addStretch()
        layout.addLayout(range_row)

        self.lbl_info = QLabel("")
        self.lbl_info.setStyleSheet("font-size: 8pt; color: #888;")
        layout.addWidget(self.lbl_info)

        layout.addStretch()

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setFixedWidth(80)
        btn_cancel.clicked.connect(self.close)
        btn_row.addWidget(btn_cancel)
        self.btn_create = QPushButton("Create")
        self.btn_create.setFixedWidth(80)
        self.btn_create.setStyleSheet(
            "QPushButton { background: #4a90e2; color: white; border: 1px solid #357abd;"
            " border-radius: 4px; font-size: 9pt; padding: 4px; }"
            "QPushButton:hover { background: #357abd; }"
        )
        self.btn_create.clicked.connect(self._on_create)
        btn_row.addWidget(self.btn_create)
        layout.addLayout(btn_row)

        self.spin_start.valueChanged.connect(self._update_info)
        self.spin_end.valueChanged.connect(self._update_info)
        self._update_info()

    def _browse(self):
        """Prompt for the parent directory used to create the new layer folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select parent directory")
        if folder:
            self.txt_path.setText(folder)
            self._update_info()

    def _update_info(self):
        """Update the info label and default folder name for the current frame range."""
        start = self.spin_start.value()
        end   = self.spin_end.value()
        if end >= start:
            count = end - start + 1
            self.lbl_info.setText(f"Will create {count} blank frame image(s).")
        else:
            self.lbl_info.setText("End index must be ≥ start index.")

        if not self._name_edited_by_user:
            self.txt_folder_name.setText(f"frame {start} - frame {end}")

    def _on_folder_name_edited(self, text):
        """Remember that the user manually edited the folder name field.

        Args:
            text (str): Current contents of the folder name field.
        """
        self._name_edited_by_user = True

    def _on_create(self):
        """Validate inputs and create a folder of blank mask images for the frame range."""
        parent_path = self.txt_path.text().strip()
        start = self.spin_start.value()
        end   = self.spin_end.value()

        if not parent_path or not os.path.isdir(parent_path):
            QMessageBox.warning(self, "Invalid path", "Please select a valid parent directory.")
            return
        if end < start:
            QMessageBox.warning(self, "Invalid range", "End index must be ≥ start index.")
            return

        folder_name = self.txt_folder_name.text().strip() or f"frame {start} - frame {end}"
        full_path = os.path.join(parent_path, folder_name)

        try:
            os.makedirs(full_path, exist_ok=True)
            if self._frame_size is not None:
                fh, fw = self._frame_size
            else:
                fh, fw = 512, 512
            for idx in range(start, end + 1):
                img_path = os.path.join(full_path, f"{idx}.png")
                if not os.path.exists(img_path):
                    blank = np.zeros((fh, fw, 3), dtype=np.uint8)
                    cv2.imwrite(img_path, blank)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create layer folder:\n{e}")
            return

        self.result_folder = full_path
        self.close()


def _normalize_frame(frame):
    """Scale a raw image array to uint8, matching the normalization used for on-screen preview.

    Args:
        frame (np.ndarray | None): Raw frame data, possibly uint16 or otherwise out-of-range.

    Returns:
        np.ndarray | None: uint8 frame, or None if the input was None.
    """
    if frame is None:
        return None
    if frame.dtype == np.uint16 or (frame.max() > 255 and frame.dtype != np.uint8):
        frame = (
            (frame.astype(np.float32) - frame.min()) /
            (frame.max() - frame.min() + 1e-6) * 255
        ).astype(np.uint8)
    else:
        frame = frame.astype(np.uint8)
    return frame


class FolderVideoSource:
    """Adapts a folder of numbered image files to the same `source[frame_idx]` indexing
    interface that propagation code expects from an h5py Dataset."""

    def __init__(self, folder_path):
        """Index every image file in the folder by the frame number found in its filename.

        Args:
            folder_path (str): Directory containing numbered frame images.
        """
        self.folder_path = folder_path
        self._index = {}
        try:
            files = sorted(os.listdir(folder_path))
        except Exception:
            files = []
        for fn in files:
            if not fn.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                continue
            m = re.search(r'(\d+)', fn)
            if m:
                self._index[int(m.group(1))] = os.path.join(folder_path, fn)

    def __getitem__(self, frame_idx):
        """Read and normalize the image file matching a given frame index.

        Args:
            frame_idx (int): Frame number to look up.

        Returns:
            np.ndarray: uint8 frame data.
        """
        path = self._index.get(frame_idx)
        if path is None:
            raise KeyError(f"No frame {frame_idx} found in folder {self.folder_path}")
        frame = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if frame is None:
            raise IOError(f"Failed to read frame {frame_idx} from {path}")
        return _normalize_frame(frame)


class TifVideoSource:
    """Adapts an in-memory list of TIFF pages to the same `source[frame_idx]` indexing
    interface that propagation code expects from an h5py Dataset."""

    def __init__(self, frames):
        """Store the decoded TIFF pages for indexed access.

        Args:
            frames (list[np.ndarray] | None): Frames decoded from a multi-page TIFF.
        """
        self.frames = frames or []

    def __getitem__(self, frame_idx):
        """Return a normalized frame by index.

        Args:
            frame_idx (int): Frame number to look up.

        Returns:
            np.ndarray: uint8 frame data.
        """
        if frame_idx < 0 or frame_idx >= len(self.frames):
            raise IndexError(f"Frame {frame_idx} out of range")
        frame = self.frames[frame_idx]
        if frame is None:
            raise IOError(f"Frame {frame_idx} is empty")
        return _normalize_frame(frame)


@contextlib.contextmanager
def _h5_video_source(h5_path, h5_key):
    """Open an HDF5 file and yield its frame dataset for the duration of the context.

    Args:
        h5_path (str): Path to the HDF5 file.
        h5_key (str): Dataset key within the file.
    """
    with h5py.File(h5_path, 'r', locking=False) as hf:
        yield hf[h5_key]


@contextlib.contextmanager
def _static_video_source(source_obj):
    """Wrap an already-constructed frame source (folder/tif) so it can be used
    with the same `with ... as video_data:` pattern as the HDF5 source.

    Args:
        source_obj: Object supporting `source_obj[frame_idx] -> np.ndarray`.
    """
    yield source_obj


class MultiLayerPropagationWorker(QObject):
    """Runs propagation for multiple layers on a background QThread"""

    batch_done  = pyqtSignal(int, object)
    layer_error = pyqtSignal(int, str)
    finished    = pyqtSignal(object)

    def __init__(self, video_source_factory, batches, layer_jobs, stop_event):
        """Store the propagation configuration to run on a background thread.

        Args:
            video_source_factory (Callable[[], ContextManager]): Zero-arg callable returning
                a context manager that yields a `video_data[frame_idx] -> np.ndarray` object.
                Works for HDF5 files, image folders, or in-memory TIFF stacks alike.
            batches (list): Sequence of frame-index batches to process in order.
            layer_jobs (list): Per-layer job dicts with setup/batch functions and kwargs.
            stop_event (threading.Event): Event used to signal early cancellation.
        """
        super().__init__()
        self._video_source_factory = video_source_factory
        self._batches    = batches
        self._jobs       = layer_jobs
        self._stop_event = stop_event

    def run(self):
        """Run setup and batched propagation for every layer job, emitting progress and errors."""
        warnings.filterwarnings(
            'ignore', message=r'`sklearn\.utils\.parallel\.delayed`',
            category=UserWarning)

        from model.feature_extraction import clear_feature_cache
        from utils.shared_utils import clear_propagation_caches
        clear_propagation_caches()
        clear_feature_cache()

        all_tl = {job['layer_idx']: {} for job in self._jobs}

        try:
            with self._video_source_factory() as video_data:
                states = {}
                for job in self._jobs:
                    try:
                        state = job['setup_fn'](video_data=video_data,
                                                **job['setup_kwargs'])
                        states[job['layer_idx']] = state
                        if state.get('start_tl'):
                            all_tl[job['layer_idx']].update(state['start_tl'])
                            self.batch_done.emit(job['layer_idx'], state['start_tl'])
                    except Exception as e:
                        states[job['layer_idx']] = None
                        logger.error("Layer %d: setup failed:\n%s",
                                    job['layer_idx'], traceback.format_exc())
                        self.layer_error.emit(
                            job['layer_idx'],
                            f"setup failed ({e.__class__.__name__}: {e}) — layer skipped for this run")

                for batch in self._batches:
                    if self._stop_event.is_set():
                        break
                    for job in self._jobs:
                        if self._stop_event.is_set():
                            break
                        state = states.get(job['layer_idx'])
                        if state is None:
                            continue
                        try:
                            batch_tl = job['batch_fn'](state=state, batch=batch,
                                                       stop_event=self._stop_event)
                            if batch_tl:
                                all_tl[job['layer_idx']].update(batch_tl)
                                self.batch_done.emit(job['layer_idx'], batch_tl)
                        except Exception as e:
                            logger.error("Layer %d: batch %s failed:\n%s",
                                        job['layer_idx'], batch, traceback.format_exc())
                            self.layer_error.emit(
                                job['layer_idx'],
                                f"batch {batch} failed ({e.__class__.__name__}: {e}) — skipped, will retry next batch")
        except Exception:
            logger.error("Propagation worker crashed:\n%s", traceback.format_exc())

        self.finished.emit(all_tl)



class PyQtMaskEditorWorkspace(QMainWindow):
    def __init__(self):
        """Initialize application state, build the UI layout, and register keyboard shortcuts."""
        super().__init__()
        self.setWindowTitle("Mask Editor")
        self.setGeometry(100, 100, 1400, 900)

        self.h5_path   = ""
        self.h5_key    = "frames"
        self.layers    = []
        self.active_layer_idx = 0

        self.edge_mode  = "upper_left"
        self.mask_alpha = 0.5
        self.video_alpha = 0.5
        self.draw_mode  = 'point'
        self.eraser_size = 5
        self.zoom_factor = 1.0

        self.bg_source_type  = None
        self.bg_folder_path  = ""
        self.bg_tif_path     = ""
        self.bg_tif_frames   = None

        self.current_video_raw = None
        self.frame_count       = 0
        self.current_idx       = 0

        self._layer_row_widgets = []
        self._current_qimage_buffer = None

        self.init_ui_layout()
        self.setFocusPolicy(Qt.StrongFocus)

        from PyQt5.QtWidgets import QShortcut
        from PyQt5.QtGui import QKeySequence

        self.shortcut_up = QShortcut(QKeySequence(Qt.Key_Up), self)
        self.shortcut_up.activated.connect(self._shortcut_layer_up)

        self.shortcut_down = QShortcut(QKeySequence(Qt.Key_Down), self)
        self.shortcut_down.activated.connect(self._shortcut_layer_down)

        self.shortcut_left = QShortcut(QKeySequence(Qt.Key_Left), self)
        self.shortcut_left.activated.connect(self._shortcut_frame_back)

        self.shortcut_right = QShortcut(QKeySequence(Qt.Key_Right), self)
        self.shortcut_right.activated.connect(self._shortcut_frame_forward)

        self.shortcut_propagate = QShortcut(QKeySequence(Qt.Key_Return), self)
        self.shortcut_propagate.activated.connect(self.apply_geometric_processing)

    def active_layer(self):
        """Get the currently selected layer.

        Returns:
            MaskLayer | None: The active layer, or None if there are no layers.
        """
        if not self.layers or self.active_layer_idx >= len(self.layers):
            return None
        return self.layers[self.active_layer_idx]

    def _color_for_new_layer(self):
        """Pick the next default color for a newly added layer.

        Returns:
            QColor: Color assigned to the next layer, cycling through the default palette.
        """
        return LAYER_DEFAULT_COLORS[len(self.layers) % len(LAYER_DEFAULT_COLORS)]

    def init_ui_layout(self):
        """Build the main window's sidebar, alpha slider, toolbar, and viewport layout."""
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        sidebar = QFrame()
        sidebar.setFixedWidth(360)
        sidebar.setFrameShape(QFrame.StyledPanel)
        sb_layout = QVBoxLayout(sidebar)
        sb_layout.setContentsMargins(10, 10, 10, 10)
        sb_layout.setAlignment(Qt.AlignTop)

        lbl_vid = QLabel("Background Source:")
        lbl_vid.setStyleSheet("font-size: 9pt; font-weight: bold;")
        sb_layout.addWidget(lbl_vid)
        h5_row = QHBoxLayout()
        self.txt_h5 = QLineEdit()
        self.txt_h5.setPlaceholderText("H5 file, image folder, or .tif video…")
        self.txt_h5.editingFinished.connect(self._handle_h5_text)
        h5_row.addWidget(self.txt_h5)
        btn_h5 = QPushButton("Browse")
        btn_h5.setFixedWidth(60)
        btn_h5.clicked.connect(self._browse_h5)
        h5_row.addWidget(btn_h5)
        sb_layout.addLayout(h5_row)

        self._add_divider(sb_layout)

        lbl_layers = QLabel("Mask Layers")
        lbl_layers.setStyleSheet("font-size: 9pt; font-weight: bold; font-style: italic;")
        sb_layout.addWidget(lbl_layers)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setMinimumHeight(200)
        self.layer_list_container = QWidget()
        self.layer_list_layout    = QVBoxLayout(self.layer_list_container)
        self.layer_list_layout.setAlignment(Qt.AlignTop)
        self.layer_list_layout.setContentsMargins(0, 0, 0, 0)
        self.layer_list_layout.setSpacing(2)
        scroll.setWidget(self.layer_list_container)
        sb_layout.addWidget(scroll)

        layer_btn_row = QHBoxLayout()
        layer_btn_row.setSpacing(4)

        btn_add_layer = QPushButton("Add layer")
        btn_add_layer.setStyleSheet("font-size: 9pt; padding: 4px;")
        btn_add_layer.clicked.connect(self._add_layer_interactive)
        layer_btn_row.addWidget(btn_add_layer)

        btn_create_layer = QPushButton("Create new layer")
        btn_create_layer.setStyleSheet("font-size: 9pt; padding: 4px;")
        btn_create_layer.clicked.connect(self._create_new_layer_interactive)
        layer_btn_row.addWidget(btn_create_layer)

        sb_layout.addLayout(layer_btn_row)

        self._add_divider(sb_layout)

        lbl_frames = QLabel("Frames")
        lbl_frames.setStyleSheet("font-size: 9pt; font-weight: bold; font-style: italic;")
        sb_layout.addWidget(lbl_frames)

        tl = QHBoxLayout()
        self.timeline_slider = QSlider(Qt.Horizontal)
        self.timeline_slider.setMinimum(1)
        self.timeline_slider.setMaximum(1)
        self.timeline_slider.setEnabled(False)
        self.timeline_slider.valueChanged.connect(self.handle_slider_scrub)
        tl.addWidget(self.timeline_slider)
        self.spin_frame = QSpinBox()
        self.spin_frame.setMinimum(1)
        self.spin_frame.setMaximum(1)
        self.spin_frame.setFixedWidth(55)
        self.spin_frame.setEnabled(False)
        self.spin_frame.valueChanged.connect(self.handle_spin_counter)
        tl.addWidget(self.spin_frame)
        sb_layout.addLayout(tl)

        self._add_divider(sb_layout)

        lbl_clip = QLabel("Clipping Options")
        lbl_clip.setStyleSheet("font-size: 9pt; font-weight: bold; font-style: italic;")
        sb_layout.addWidget(lbl_clip)

        self.combo_edge = QComboBox()
        self.combo_edge.addItems([
            "Upper/Left edge align  [1]",
            "Lower/Right edge align [2]",
            "Hole fill              [3]",
            "Hole crop              [4]",
            "Object                 [5]",
        ])
        self.combo_edge.currentIndexChanged.connect(self.change_edge_mode)
        sb_layout.addWidget(self.combo_edge)

        self.btn_clear = QPushButton("Clear current guideline")
        self.btn_clear.setEnabled(False)
        self.btn_clear.clicked.connect(self.clear_drawing)
        sb_layout.addWidget(self.btn_clear)

        self.btn_paste_line = QPushButton("Paste previous guideline")
        self.btn_paste_line.setEnabled(False)
        self.btn_paste_line.clicked.connect(self.get_prev_line)
        sb_layout.addWidget(self.btn_paste_line)

        self.btn_process = QPushButton("Process corrections")
        self.btn_process.setEnabled(False)
        self.btn_process.clicked.connect(self.apply_geometric_processing)
        sb_layout.addWidget(self.btn_process)

        self._add_divider(sb_layout)

        lbl_propagator = QLabel("Propagation")
        lbl_propagator.setStyleSheet("font-size: 9pt; font-weight: bold; font-style: italic;")
        sb_layout.addWidget(lbl_propagator)

        self.prop_container = QFrame()
        self.prop_container.setFrameShape(QFrame.StyledPanel)
        self.prop_container.setStyleSheet(
            "QFrame { border: 1px solid #e0e0e0; border-radius: 4px; background: #fafafa; }"
        )
        self.prop_container.setEnabled(False)
        prop_outer = QVBoxLayout(self.prop_container)
        prop_outer.setContentsMargins(8, 6, 8, 6)
        prop_outer.setSpacing(4)

        frames_row = QHBoxLayout()
        lbl_steps  = QLabel("Frames:")
        lbl_steps.setStyleSheet("font-size: 9pt;")
        frames_row.addWidget(lbl_steps)
        self.spin_prop_steps = QSpinBox()
        self.spin_prop_steps.setMinimum(1)
        self.spin_prop_steps.setMaximum(9999)
        self.spin_prop_steps.setValue(20)
        self.spin_prop_steps.setFixedWidth(65)
        self.spin_prop_steps.setToolTip("Number of frames to propagate forward (shared across all layers)")
        frames_row.addWidget(self.spin_prop_steps)
        frames_row.addStretch()
        prop_outer.addLayout(frames_row)

        sb_layout.addWidget(self.prop_container)

        prop_btn_row = QHBoxLayout()
        self.btn_prop = QPushButton("Propagate")
        self.btn_prop.clicked.connect(self.trigger_external_propagation)
        prop_btn_row.addWidget(self.btn_prop)

        self.btn_stop_prop = QPushButton("Stop")
        self.btn_stop_prop.clicked.connect(self.stop_propagation)
        self.btn_stop_prop.setEnabled(False)
        prop_btn_row.addWidget(self.btn_stop_prop)
        sb_layout.addLayout(prop_btn_row)

        root.addWidget(sidebar)

        alpha_frame = QFrame()
        alpha_frame.setFixedWidth(55)
        alpha_layout = QVBoxLayout(alpha_frame)
        alpha_layout.setContentsMargins(5, 10, 5, 10)
        alpha_layout.setAlignment(Qt.AlignCenter)
        lbl_a_top = QLabel("Mask\n100%")
        lbl_a_top.setAlignment(Qt.AlignCenter)
        lbl_a_top.setStyleSheet("font-size: 9pt")
        alpha_layout.addWidget(lbl_a_top)
        self.lbl_alpha_mask = lbl_a_top
        self.alpha_slider = QSlider(Qt.Vertical)
        self.alpha_slider.setMinimum(0)
        self.alpha_slider.setMaximum(100)
        self.alpha_slider.setValue(50)
        self.alpha_slider.setTickPosition(QSlider.TicksBothSides)
        self.alpha_slider.setTickInterval(10)
        self.alpha_slider.valueChanged.connect(self._change_alpha)
        alpha_layout.addWidget(self.alpha_slider)
        lbl_a_bot = QLabel("Video\n100%")
        lbl_a_bot.setAlignment(Qt.AlignCenter)
        lbl_a_bot.setStyleSheet("font-size: 8pt; font-weight: bold;")
        alpha_layout.addWidget(lbl_a_bot)
        self.lbl_alpha_video = lbl_a_bot
        root.addWidget(alpha_frame)

        vp_col = QWidget()
        vp_layout = QVBoxLayout(vp_col)
        vp_layout.setContentsMargins(0, 0, 0, 0)
        vp_layout.setSpacing(4)

        toolbar = self._build_toolbar()
        vp_layout.addWidget(toolbar)

        self.lbl_filename_header = QLabel("Add layers and load masks to begin")
        self.lbl_filename_header.setStyleSheet("font-size: 9pt; font-style: italic; font-weight: bold;")
        vp_layout.addWidget(self.lbl_filename_header)

        self.view = ViewportCanvas(self)
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        vp_layout.addWidget(self.view, stretch=1)

        self.statusBar = QLabel("System Status: Ready")
        self.statusBar.setStyleSheet("font-size: 8pt; color: #555; padding-left: 5px;")
        vp_layout.addWidget(self.statusBar)

        root.addWidget(vp_col, stretch=1)

    def _build_toolbar(self):
        """Build the toolbar with draw-mode buttons, zoom controls, and undo/redo buttons.

        Returns:
            QFrame: The assembled toolbar widget.
        """
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setFixedHeight(38)
        frame.setStyleSheet("QFrame { background: white; }")
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layout = QHBoxLayout(frame)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(10)
        btn_style = (
            "QPushButton {"
            "    font-size: 9pt;"
            "    border: 1px solid #dcdcdc;"
            "    border-radius: 4px;"
            "    background-color: #fafafa;"
            "    color: #333;"
            "}"
            "QPushButton:hover {"
            "    background-color: #f0f0f0;"
            "}"
            "QPushButton:checked {"
            "    background-color: #b0b0b0;"
            "    font-weight: bold;"
            "    border-color: #999;"
            "}"
            "QPushButton:disabled {"
            "    color: #ccc;"
            "    background-color: #f5f5f5;"
            "    border-color: #e0e0e0;"
            "}"
        )

        self.btn_mode_point  = QPushButton("Point [Q]")
        self.btn_mode_brush  = QPushButton("Brush [W]")
        self.btn_mode_eraser = QPushButton("Eraser [E]")

        for btn in (self.btn_mode_point, self.btn_mode_brush, self.btn_mode_eraser):
            btn.setCheckable(True)
            btn.setStyleSheet(btn_style)
            btn.setFixedHeight(26)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            layout.addWidget(btn, stretch=1)

        self.btn_mode_point.setChecked(True)
        self.btn_mode_point.clicked.connect(lambda: self.set_draw_mode('point'))
        self.btn_mode_brush.clicked.connect(lambda: self.set_draw_mode('brush'))
        self.btn_mode_eraser.clicked.connect(lambda: self.set_draw_mode('eraser'))
        self._mode_buttons = [self.btn_mode_point, self.btn_mode_brush, self.btn_mode_eraser]

        self._add_toolbar_sep(layout)

        lbl_zoom = QLabel("Zoom:")
        lbl_zoom.setStyleSheet("font-size: 9pt; border: none;")
        layout.addWidget(lbl_zoom)

        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(10)
        self.zoom_slider.setMaximum(500)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.zoom_slider.setMinimumWidth(80)
        self.zoom_slider.setFixedHeight(26)
        self.zoom_slider.valueChanged.connect(self._update_zoom)
        layout.addWidget(self.zoom_slider, stretch=2)

        self.lbl_zoom_val = QLabel("1.0×")
        self.lbl_zoom_val.setStyleSheet("font-size: 9pt; min-width:34px; border: none;")
        layout.addWidget(self.lbl_zoom_val)

        btn_zoom_reset = QPushButton("Fit")
        btn_zoom_reset.setStyleSheet(btn_style)
        btn_zoom_reset.setFixedHeight(26)
        btn_zoom_reset.setFixedWidth(32)
        btn_zoom_reset.clicked.connect(self._reset_zoom)
        layout.addWidget(btn_zoom_reset)

        self._add_toolbar_sep(layout)

        undoredo_layout = QHBoxLayout()
        undoredo_layout.setContentsMargins(0, 0, 0, 0)
        undoredo_layout.setSpacing(12)

        self.btn_undo = QPushButton("Undo")
        self.btn_undo.setEnabled(False)
        self.btn_undo.setStyleSheet(btn_style)
        self.btn_undo.setFixedHeight(26)
        self.btn_undo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_undo.clicked.connect(self.execute_undo_history_pop)

        self.btn_redo = QPushButton("Redo")
        self.btn_redo.setEnabled(False)
        self.btn_redo.setStyleSheet(btn_style)
        self.btn_redo.setFixedHeight(26)
        self.btn_redo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_redo.clicked.connect(self.execute_redo_history_pop)

        undoredo_layout.addWidget(self.btn_undo, stretch=1)
        undoredo_layout.addWidget(self.btn_redo, stretch=1)

        layout.addLayout(undoredo_layout, stretch=1)

        return frame

    def _add_toolbar_sep(self, layout):
        """Add a vertical separator with spacing to a toolbar layout.

        Args:
            layout (QLayout): Layout to append the separator to.
        """
        layout.addSpacing(8)
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep)
        layout.addSpacing(8)

    def _add_divider(self, layout):
        """Add a horizontal divider line to a sidebar layout.

        Args:
            layout (QLayout): Layout to append the divider to.
        """
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("color: #e0e0e0; margin: 8px 0;")
        layout.addWidget(line)

    def keyPressEvent(self, event):
        """Handle global keyboard shortcuts for edge mode, draw mode, and processing.

        Args:
            event (QKeyEvent): Key press event to handle.
        """
        if event.key() in EDGE_MODE_KEYS:
            new_mode = EDGE_MODE_KEYS[event.key()]
            self.edge_mode = new_mode
            combo_idx = EDGE_MODE_COMBO_INDEX[new_mode]
            self.combo_edge.blockSignals(True)
            self.combo_edge.setCurrentIndex(combo_idx)
            self.combo_edge.blockSignals(False)
            self.statusBar.setText(f"System Status: Processing mode → {new_mode.replace('_', ' ').title()}")
            self.view.redraw_guidelines()
            for row in self._layer_row_widgets:
                row._sync_prop_param_stack()
            event.accept()
            return

        if event.key() == Qt.Key_Q:
            self.set_draw_mode('point')
            event.accept()
            return
        elif event.key() == Qt.Key_W:
            self.set_draw_mode('brush')
            event.accept()
            return
        elif event.key() == Qt.Key_E:
            self.set_draw_mode('eraser')
            event.accept()
            return

        if event.key() == Qt.Key_Space:
            if self.btn_process.isEnabled():
                self.apply_geometric_processing()
                event.accept()
                return

        super().keyPressEvent(event)

    def set_active_layer_by_index(self, idx):
        """Set the active layer by index and refresh the layer list, guidelines, and history buttons.

        Args:
            idx (int): Index of the layer to activate.
        """
        if 0 <= idx < len(self.layers):
            self.active_layer_idx = idx
            self._rebuild_layer_list_ui()
            self.view.redraw_guidelines()
            self.update_history_button_states()
    def _shortcut_layer_up(self):
        """Move the active layer selection up by one, if possible."""
        if self.layers and self.active_layer_idx > 0:
            self.set_active_layer_by_index(self.active_layer_idx - 1)
            self.statusBar.setText(f"System Status: Switched to Layer {self.active_layer_idx + 1}")

    def _shortcut_layer_down(self):
        """Move the active layer selection down by one, if possible."""
        if self.layers and self.active_layer_idx < len(self.layers) - 1:
            self.set_active_layer_by_index(self.active_layer_idx + 1)
            self.statusBar.setText(f"System Status: Switched to Layer {self.active_layer_idx + 1}")

    def _shortcut_frame_back(self):
        """Step the timeline back by one frame, if possible."""
        if self.current_idx > self.timeline_slider.minimum():
            self.timeline_slider.setValue(self.current_idx - 1)
            self.statusBar.setText(f"System Status: Frame {self.current_idx}")

    def _shortcut_frame_forward(self):
        """Step the timeline forward by one frame, if possible."""
        if self.current_idx < self.timeline_slider.maximum():
            self.timeline_slider.setValue(self.current_idx + 1)
            self.statusBar.setText(f"System Status: Frame {self.current_idx}")

    def _add_layer_interactive(self):
        """Prompt for input/output folders and add a new mask layer from existing files."""
        folder = QFileDialog.getExistingDirectory(self, "Select mask input folder for new layer")
        if not folder:
            return
        out_folder = QFileDialog.getExistingDirectory(self, "Select output folder for new layer")
        if not out_folder:
            out_folder = folder

        layer = MaskLayer(folder, out_folder, self._color_for_new_layer())
        self.layers.append(layer)
        self._rebuild_layer_list_ui()
        self.reload_layer(len(self.layers) - 1)

    def _get_bg_frame_size(self):
        """Get the pixel dimensions of the current background frame, if one is loaded.

        Returns:
            tuple | None: (height, width) of the background frame, or None if no
                source is loaded.
        """
        frame = self._load_bg_frame(self.current_idx)
        if frame is not None:
            return frame.shape[:2]
        return None

    def _create_new_layer_interactive(self):
        """Open the create-layer dialog to generate blank masks, then add the result as a new layer."""
        dlg = CreateLayerDialog(self, frame_size=self._get_bg_frame_size())
        dlg.setWindowModality(Qt.ApplicationModal)
        dlg.show()
        from PyQt5.QtWidgets import QApplication as _QApp
        while dlg.isVisible():
            _QApp.processEvents()

        folder = dlg.result_folder
        if not folder or not os.path.isdir(folder):
            return

        out_folder = QFileDialog.getExistingDirectory(self, "Select output folder for new layer")
        if not out_folder:
            out_folder = folder

        layer = MaskLayer(folder, out_folder, self._color_for_new_layer())
        self.layers.append(layer)
        self._rebuild_layer_list_ui()
        self.reload_layer(len(self.layers) - 1)

    def remove_layer(self, idx):
        """Remove the layer at the given index and refresh the UI and frame range.

        Args:
            idx (int): Index of the layer to remove.
        """
        if idx < 0 or idx >= len(self.layers):
            return
        self.layers.pop(idx)
        if self.active_layer_idx >= len(self.layers):
            self.active_layer_idx = max(0, len(self.layers) - 1)
        self._rebuild_layer_list_ui()
        self._sync_frame_range()
        self.render_current_workspace_view()

    def _rebuild_layer_list_ui(self):
        """Rebuild the sidebar's layer row widgets to match the current layer list."""
        for w in self._layer_row_widgets:
            self.layer_list_layout.removeWidget(w)
            w.setParent(None)
        self._layer_row_widgets.clear()

        for i, layer in enumerate(self.layers):
            row = LayerRowWidget(i, layer, self)
            self.layer_list_layout.addWidget(row)
            self._layer_row_widgets.append(row)

    def _active_layer_changed(self, idx):
        """Set the active layer by index and refresh dependent UI state.

        Args:
            idx (int): Index of the layer that became active.
        """
        if 0 <= idx < len(self.layers):
            self.active_layer_idx = idx
            self._rebuild_layer_list_ui()
            self.view.redraw_guidelines()
            self.update_history_button_states()

    def reload_layer(self, idx):
        """Rescan a layer's input folder for mask files and mirror missing files to its output folder.

        Args:
            idx (int): Index of the layer to reload.
        """
        if idx < 0 or idx >= len(self.layers):
            return
        layer = self.layers[idx]
        if not layer.input_folder or not os.path.isdir(layer.input_folder):
            return

        all_files = sorted([
            f for f in os.listdir(layer.input_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
        ])
        if not all_files:
            layer.mask_files = []
            return

        layer.mask_files = all_files

        if layer.output_folder and not os.path.exists(layer.output_folder):
            try:
                os.makedirs(layer.output_folder, exist_ok=True)
            except Exception:
                return

        if layer.output_folder:
            for fn in layer.mask_files:
                out_p = os.path.join(layer.output_folder, fn)
                if not os.path.exists(out_p):
                    src = cv2.imread(os.path.join(layer.input_folder, fn), cv2.IMREAD_COLOR)
                    if src is not None:
                        cv2.imwrite(out_p, src)

        self._sync_frame_range()
        self.render_current_workspace_view()

    def _get_layer_frame_numbers(self, layer):
        """Extract the set of frame indices present in a layer's mask filenames.

        Args:
            layer (MaskLayer): Layer whose mask filenames are scanned.

        Returns:
            set: Frame indices parsed from the layer's mask filenames.
        """
        frames = set()
        for f in layer.mask_files:
            match = re.search(r'(\d+)', f)
            if match:
                frames.add(int(match.group(1)))
        return frames

    def _sync_frame_range(self):
        """Recompute the shared frame range across all layers and update timeline controls."""
        valid_layers = [l for l in self.layers if l.mask_files]
        if not valid_layers:
            self.frame_count = 0
            self.timeline_slider.setEnabled(False)
            self.spin_frame.setEnabled(False)
            self.btn_clear.setEnabled(False)
            self.btn_paste_line.setEnabled(False)
            self.btn_process.setEnabled(False)
            return

        common_frames = self._get_layer_frame_numbers(valid_layers[0])
        for layer in valid_layers[1:]:
            common_frames.intersection_update(self._get_layer_frame_numbers(layer))

        if not common_frames:
            return

        min_frame = min(common_frames)
        max_frame = max(common_frames)

        self.timeline_slider.blockSignals(True)
        self.spin_frame.blockSignals(True)

        self.timeline_slider.setMinimum(min_frame)
        self.timeline_slider.setMaximum(max_frame)
        self.spin_frame.setMinimum(min_frame)
        self.spin_frame.setMaximum(max_frame)

        if self.current_idx < min_frame or self.current_idx > max_frame:
            self.current_idx = min_frame

        self.timeline_slider.setValue(self.current_idx)
        self.spin_frame.setValue(self.current_idx)

        self.timeline_slider.blockSignals(False)
        self.spin_frame.blockSignals(False)

        self.timeline_slider.setEnabled(True)
        self.spin_frame.setEnabled(True)
        self.btn_clear.setEnabled(True)
        self.btn_paste_line.setEnabled(True)
        self.btn_process.setEnabled(True)

    def handle_slider_scrub(self, val):
        """Update the current frame and view when the timeline slider is scrubbed.

        Args:
            val (int): New frame index selected on the slider.
        """
        if val != self.current_idx:
            self.current_idx = val
            self.spin_frame.blockSignals(True)
            self.spin_frame.setValue(val)
            self.spin_frame.blockSignals(False)

            self._load_guideline_for_current_frame()

            self.render_current_workspace_view()

    def handle_spin_counter(self, val):
        """Update the current frame and view when the frame spin box value changes.

        Args:
            val (int): New frame index entered in the spin box.
        """
        if val != self.current_idx:
            self.current_idx = val
            self.timeline_slider.blockSignals(True)
            self.timeline_slider.setValue(val)
            self.timeline_slider.blockSignals(False)

            self._load_guideline_for_current_frame()

            self.render_current_workspace_view()

    def _load_guideline_for_current_frame(self):
        """Restore each layer's drawn strokes for the current frame, or clear them if none were saved."""
        for layer in self.layers:
            if self.current_idx in layer.frame_guidelines:
                layer.strokes = copy.deepcopy(layer.frame_guidelines[self.current_idx])
            else:
                layer.strokes = []

    def _find_file_by_frame(self, layer, frame_num):
        """Locate a layer's mask filename matching a given frame number.

        Args:
            layer (MaskLayer): Layer whose mask files are searched.
            frame_num (int): Frame index to match against filenames.

        Returns:
            tuple: (full_path, filename) if found, otherwise (None, None).
        """
        for fn in layer.mask_files:
            match = re.search(r'(\d+)', fn)
            if match and int(match.group(1)) == frame_num:
                folder = layer.output_folder or layer.input_folder
                return os.path.join(folder, fn), fn
        return None, None

    def _browse_h5(self):
        """Prompt the user to choose an HDF5 file, TIFF file, or image folder as the background source."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Background",
            "",
            "Supported files (*.h5 *.hdf5 *.tif *.tiff);;HDF5 Files (*.h5 *.hdf5);;TIFF Files (*.tif *.tiff);;All Files (*)"
        )
        if path:
            self._apply_bg_path(path)
            return
        folder = QFileDialog.getExistingDirectory(self, "…or select an image folder as background")
        if folder:
            self._apply_bg_path(folder)

    def _handle_h5_text(self):
        """Apply the background source path typed directly into the text field."""
        path = self.txt_h5.text().strip()
        if path:
            self._apply_bg_path(path)

    def _apply_bg_path(self, path):
        """Determine the background source type from a path and store it for frame loading.

        Args:
            path (str): Path to an image folder, .h5/.hdf5 file, or .tif/.tiff file.
        """
        self.bg_tif_frames = None
        if os.path.isdir(path):
            self.bg_source_type = 'folder'
            self.bg_folder_path = path
            self.h5_path = ""
            self.bg_tif_path = ""
        elif os.path.isfile(path):
            ext = os.path.splitext(path)[1].lower()
            if ext in ('.tif', '.tiff'):
                self.bg_source_type = 'tif'
                self.bg_tif_path = path
                self.h5_path = ""
                self.bg_folder_path = ""
            elif ext in ('.h5', '.hdf5'):
                self.bg_source_type = 'h5'
                self.h5_path = path
                self.bg_tif_path = ""
                self.bg_folder_path = ""
            else:
                return
        else:
            return
        self.txt_h5.setText(path)
        self.render_current_workspace_view()

    def _load_bg_frame(self, frame_idx):
        """Load and normalize a single background frame from the active source.

        Args:
            frame_idx (int): Index of the frame to load.

        Returns:
            np.ndarray | None: uint8 grayscale or RGB frame, or None if unavailable.
        """
        if self.bg_source_type == 'h5':
            if not self.h5_path or not os.path.exists(self.h5_path):
                return None
            try:
                with h5py.File(self.h5_path, 'r', locking=False) as f:
                    raw_v = f[self.h5_key][frame_idx]
                    if raw_v.dtype == np.uint16 or np.max(raw_v) > 255:
                        return (
                            (raw_v - np.min(raw_v)) /
                            (np.max(raw_v) - np.min(raw_v) + 1e-6) * 255
                        ).astype(np.uint8)
                    return raw_v.astype(np.uint8)
            except Exception:
                return None

        elif self.bg_source_type == 'tif':
            if not self.bg_tif_path or not os.path.exists(self.bg_tif_path):
                return None
            if self.bg_tif_frames is None:
                try:
                    ret, frames = cv2.imreadmulti(self.bg_tif_path, flags=cv2.IMREAD_UNCHANGED)
                    self.bg_tif_frames = frames if ret and frames else []
                except Exception:
                    self.bg_tif_frames = []
            if not self.bg_tif_frames or frame_idx >= len(self.bg_tif_frames):
                return None
            frame = self.bg_tif_frames[frame_idx]
            if frame is None:
                return None
            if frame.dtype == np.uint16 or (frame.max() > 255 and frame.dtype != np.uint8):
                frame = (
                    (frame.astype(np.float32) - frame.min()) /
                    (frame.max() - frame.min() + 1e-6) * 255
                ).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
            return frame

        elif self.bg_source_type == 'folder':
            if not self.bg_folder_path or not os.path.isdir(self.bg_folder_path):
                return None
            try:
                files = sorted(os.listdir(self.bg_folder_path))
            except Exception:
                return None
            for fn in files:
                if not fn.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                    continue
                m = re.search(r'(\d+)', fn)
                if m and int(m.group(1)) == frame_idx:
                    img_path = os.path.join(self.bg_folder_path, fn)
                    frame = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                    if frame is None:
                        return None
                    if frame.dtype == np.uint16 or (frame.max() > 255):
                        frame = (
                            (frame.astype(np.float32) - frame.min()) /
                            (frame.max() - frame.min() + 1e-6) * 255
                        ).astype(np.uint8)
                    else:
                        frame = frame.astype(np.uint8)
                    return frame
            return None

        return None

    def set_draw_mode(self, mode):
        """Switch the active drawing tool and update the toolbar buttons and cursor.

        Args:
            mode (str): Drawing mode to activate ('point', 'brush', or 'eraser').
        """
        self.draw_mode = mode
        for btn in self._mode_buttons:
            btn.setChecked(False)
        if mode == 'point':
            self.btn_mode_point.setChecked(True)
            self.view.setCursor(Qt.ArrowCursor)
        elif mode == 'brush':
            self.btn_mode_brush.setChecked(True)
            self.view.setCursor(Qt.CrossCursor)
        elif mode == 'eraser':
            self.btn_mode_eraser.setChecked(True)
            self.view.setCursor(Qt.CrossCursor)

    def _update_zoom(self, val):
        """Apply a zoom percentage to the viewport canvas.

        Args:
            val (int): Zoom level as a percentage (100 = fit-to-view).
        """
        self.zoom_factor = val / 100.0
        self.lbl_zoom_val.setText(f"{self.zoom_factor:.1f}×")
        if self.view.pixmap_item.pixmap() and not self.view.pixmap_item.pixmap().isNull():
            pm = self.view.pixmap_item.pixmap()
            self.view.resetTransform()
            if val == 100:
                self.view.fitInView(self.view.pixmap_item, Qt.KeepAspectRatio)
            else:
                rect = self.view.viewport().rect()
                img_w = pm.width()
                img_h = pm.height()
                if img_w > 0 and img_h > 0:
                    base_scale = min(rect.width() / img_w, rect.height() / img_h)
                    scale = base_scale * self.zoom_factor
                    self.view.resetTransform()
                    self.view.scale(scale, scale)

    def _reset_zoom(self):
        """Reset the zoom slider to 100% (fit-to-view)."""
        self.zoom_slider.setValue(100)

    def save_stroke_history_for_frame(self, layer):
        """Buffer the layer's current strokes for reuse and refresh dependent UI state.

        Args:
            layer (MaskLayer): Layer whose strokes were just modified.
        """
        if layer.strokes:
            layer.last_drawn_strokes_buffer = copy.deepcopy(layer.strokes)
        has_stroke = bool(layer.strokes)
        self.prop_container.setEnabled(has_stroke)
        self.update_history_button_states()
        for row in self._layer_row_widgets:
            if row.layer is layer:
                row._sync_prop_param_stack()
                break

    def update_history_button_states(self):
        """Enable or disable the undo/redo buttons based on history for the current frame."""
        layer = self.active_layer()
        if layer is None or not layer.mask_files:
            self.btn_undo.setEnabled(False)
            self.btn_redo.setEnabled(False)
            return
        _, fn = self._find_file_by_frame(layer, self.current_idx)
        if not fn:
            self.btn_undo.setEnabled(False)
            self.btn_redo.setEnabled(False)
            return
        self.btn_undo.setEnabled(bool(layer.undo_stacks.get(fn)))
        self.btn_redo.setEnabled(bool(layer.redo_stacks.get(fn)))

    def push_state_to_undo_stack(self, layer, filename, mask_matrix):
        """Push a copy of a mask onto a layer's per-file undo stack.

        Args:
            layer (MaskLayer): Layer whose undo stack is updated.
            filename (str): Mask filename the snapshot belongs to.
            mask_matrix (np.ndarray): Binary mask array to snapshot.
        """
        if filename not in layer.undo_stacks:
            layer.undo_stacks[filename] = []
        layer.undo_stacks[filename].append(mask_matrix.copy())

    def _snapshot_layer_frames_for_undo(self, layer, frame_indices):
        """Capture each frame's current on-disk mask into the undo stack before propagation overwrites it.

        Args:
            layer (MaskLayer): Layer whose frames are snapshotted.
            frame_indices (iterable): Frame indices to capture.
        """
        for f in frame_indices:
            path, fn = self._find_file_by_frame(layer, f)
            if not path or not fn:
                continue
            raw_img = cv2.imread(path, cv2.IMREAD_COLOR)
            if raw_img is None:
                continue
            mask_bin = self._extract_binary_from_color_image(raw_img, layer.color)
            self.push_state_to_undo_stack(layer, fn, mask_bin)
            layer.redo_stacks.pop(fn, None)

    def _extract_binary_from_color_image(self, bgr_img, qt_color):
        """Convert a color-encoded mask image back into a binary mask matching a given color.

        Args:
            bgr_img (np.ndarray): Mask image in BGR color order.
            qt_color (QColor): Color used to encode foreground pixels.

        Returns:
            np.ndarray: Binary mask (0 or 255) where pixels matched the target color.
        """
        b = qt_color.blue()
        g = qt_color.green()
        r = qt_color.red()

        if len(bgr_img.shape) == 3 and bgr_img.shape[2] == 3:
            diff = np.abs(bgr_img.astype(np.int16) - np.array([b, g, r], dtype=np.int16))
            mask_bin = (np.max(diff, axis=2) < 20).astype(np.uint8) * 255
            if np.sum(mask_bin) == 0:
                gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
                _, mask_bin = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            return mask_bin
        else:
            _, mask_bin = cv2.threshold(bgr_img, 1, 255, cv2.THRESH_BINARY)
            return mask_bin


    def render_current_workspace_view(self):
        """Composite the background frame with all visible layer masks and update the viewport."""
        if not self.layers:
            self.view.pixmap_item.setPixmap(QPixmap())
            self.lbl_filename_header.setText("Add layers and load masks to begin")
            return

        self.current_video_raw = self._load_bg_frame(self.current_idx)

        if self.current_video_raw is not None:
            h, w = self.current_video_raw.shape[:2]
        else:
            h, w = None, None
            for layer in self.layers:
                path, _ = self._find_file_by_frame(layer, self.current_idx)
                if path:
                    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if m is not None:
                        h, w = m.shape
                        break
            if h is None:
                return

        if self.current_video_raw is not None:
            vid = self.current_video_raw
            if vid.ndim == 2:
                base_rgb = cv2.cvtColor(vid, cv2.COLOR_GRAY2RGB).astype(np.float32)
            elif vid.shape[2] == 4:
                base_rgb = vid[:, :, :3].astype(np.float32)
            else:
                base_rgb = vid.astype(np.float32)
            base_rgb = base_rgb * self.video_alpha
        else:
            base_rgb = np.zeros((h, w, 3), dtype=np.float32)

        for layer in self.layers:
            if not layer.visible or not layer.mask_files:
                continue
            path, _ = self._find_file_by_frame(layer, self.current_idx)
            if not path:
                continue
            raw_img = cv2.imread(path, cv2.IMREAD_COLOR)
            if raw_img is None:
                continue

            if raw_img.shape[:2] != (h, w):
                raw_img = cv2.resize(raw_img, (w, h), interpolation=cv2.INTER_NEAREST)

            mask_bin = self._extract_binary_from_color_image(raw_img, layer.color)

            if np.sum(mask_bin) == 0:
                layer.current_mask_raw = mask_bin
                continue

            layer.current_mask_raw = mask_bin

            c = layer.color
            alpha = c.alpha() / 255.0 * self.mask_alpha
            tint = np.array([c.red(), c.green(), c.blue()], dtype=np.float32)
            mask_norm = (mask_bin > 0).astype(np.float32)
            for ch in range(3):
                base_rgb[:, :, ch] = (
                    base_rgb[:, :, ch] * (1 - mask_norm * alpha) +
                    tint[ch] * mask_norm * alpha
                )

        al = self.active_layer()
        if al is not None:
            path, _ = self._find_file_by_frame(al, self.current_idx)
            if path:
                raw_img = cv2.imread(path, cv2.IMREAD_COLOR)
                if raw_img is not None:
                    if raw_img.shape[:2] != (h, w):
                        raw_img = cv2.resize(raw_img, (w, h), interpolation=cv2.INTER_NEAREST)
                    al.current_mask_raw = self._extract_binary_from_color_image(raw_img, al.color)

        display_rgb = np.clip(base_rgb, 0, 255).astype(np.uint8)

        self._current_qimage_buffer = display_rgb
        q_img = QImage(self._current_qimage_buffer.data, w, h, 3 * w, QImage.Format_RGB888)

        self.view.pixmap_item.setPixmap(QPixmap.fromImage(q_img))
        self.view.setSceneRect(0, 0, w, h)
        if hasattr(self, 'zoom_slider') and self.zoom_slider.value() != 100:
            self._update_zoom(self.zoom_slider.value())
        else:
            self.view.fitInView(self.view.pixmap_item, Qt.KeepAspectRatio)
        self.view.redraw_guidelines()

        names = []
        for i, layer in enumerate(self.layers):
            _, fn = self._find_file_by_frame(layer, self.current_idx)
            if fn:
                names.append(f"L{i+1}:{fn}")
        self.lbl_filename_header.setText("  |  ".join(names) if names else "—")

        self.update_history_button_states()


    def apply_geometric_processing(self):
        """Apply each layer's drawn guideline strokes to its mask for the current frame, grouped by edge mode."""
        processed_count = 0

        for layer_idx, layer in enumerate(self.layers):
            if not layer.mask_files or layer.current_mask_raw is None or not layer.strokes:
                continue

            strokes_by_mode = {}
            for stroke in layer.strokes:
                mode = stroke.get('edge_mode', self.edge_mode)
                strokes_by_mode.setdefault(mode, []).append(stroke)

            _, fn = self._find_file_by_frame(layer, self.current_idx)
            if not fn:
                continue

            self.push_state_to_undo_stack(layer, fn, layer.current_mask_raw)
            layer.redo_stacks.pop(fn, None)

            for stroke_mode, strokes in strokes_by_mode.items():
                all_pts = []
                for s in strokes:
                    all_pts.extend(s['pts'])

                if stroke_mode in ["hole_fill", "hole_crop", "object"]:
                    if len(all_pts) < 3:
                        continue
                else:
                    if len(all_pts) < 2:
                        continue

                saved_mode = self.edge_mode
                self.edge_mode = stroke_mode

                if stroke_mode == "hole_fill":
                    self._exec_hole_fill(layer, all_pts)
                elif stroke_mode == "hole_crop":
                    self._exec_hole_crop(layer, all_pts)
                elif stroke_mode == "object":
                    self._exec_object_mode(layer, all_pts)
                else:
                    self._exec_coordinate_clamp(layer, all_pts)

                self.edge_mode = saved_mode

            processed_count += 1

        if processed_count > 0:
            self.statusBar.setText(f"System Status: Batch processing complete. Modified {processed_count} layer masks.")
            self.render_current_workspace_view()
        else:
            self.statusBar.setText("System Status: Processing skipped. No valid guidelines found on visible layers.")

    def _save_layer_mask(self, layer, mask_binary):
        """Write a binary mask to disk in the layer's color encoding and clear its strokes.

        Args:
            layer (MaskLayer): Layer whose mask file is written.
            mask_binary (np.ndarray): Binary mask (0 or 255) to save.
        """
        path, fn = self._find_file_by_frame(layer, self.current_idx)
        if path:
            h, w = mask_binary.shape
            color_mask = np.zeros((h, w, 3), dtype=np.uint8)
            color_mask[mask_binary > 0] = [layer.color.blue(), layer.color.green(), layer.color.red()]
            cv2.imwrite(path, color_mask)

        layer.strokes.clear()
        self.save_stroke_history_for_frame(layer)

    def _exec_hole_fill(self, layer, pts):
        """Fill the polygon enclosed by guideline points into the layer's mask and save it.

        Args:
            layer (MaskLayer): Layer being edited.
            pts (list): Guideline points defining the polygon to fill.
        """
        mat = layer.current_mask_raw.copy()
        cv2.fillPoly(mat, [np.array(pts, dtype=np.int32).reshape(-1, 1, 2)], 255)
        _, final = cv2.threshold(mat, 127, 255, cv2.THRESH_BINARY)
        self._save_layer_mask(layer, final)

    def _exec_hole_crop(self, layer, pts):
        """Clear the polygon enclosed by guideline points from the layer's mask and save it.

        Args:
            layer (MaskLayer): Layer being edited.
            pts (list): Guideline points defining the polygon to remove.
        """
        mat = layer.current_mask_raw.copy()
        crop = np.full_like(mat, 255)
        cv2.fillPoly(crop, [np.array(pts, dtype=np.int32).reshape(-1, 1, 2)], 0)
        mat = cv2.bitwise_and(mat, crop)
        _, final = cv2.threshold(mat, 127, 255, cv2.THRESH_BINARY)
        self._save_layer_mask(layer, final)

    def _exec_object_mode(self, layer, pts):
        """Replace the layer's mask with a fresh polygon defined by guideline points and save it.

        Args:
            layer (MaskLayer): Layer being edited.
            pts (list): Guideline points defining the object polygon.
        """
        h, w = layer.current_mask_raw.shape
        obj_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(obj_mask, [np.array(pts, dtype=np.int32).reshape(-1, 1, 2)], 255)
        _, final = cv2.threshold(obj_mask, 127, 255, cv2.THRESH_BINARY)
        self._save_layer_mask(layer, final)

    def _exec_coordinate_clamp(self, layer, pts):
        """Clip or extend the mask along a drawn guideline according to the current edge mode, then save it.

        Args:
            layer (MaskLayer): Layer being edited.
            pts (list): Guideline points defining the edge line.
        """
        mask_mat = layer.current_mask_raw.copy()
        np_pts = np.array(pts, dtype=np.int32)
        h, w = mask_mat.shape

        guide_canvas = np.zeros((h, w), dtype=np.uint8)
        cv2.polylines(guide_canvas, [np_pts], False, 1, thickness=1)

        for x in range(w):
            l_ys = np.where(guide_canvas[:, x] > 0)[0]
            if len(l_ys) > 0:
                y_line = int(np.mean(l_ys))
                if self.edge_mode == 'upper_left':
                    mask_mat[0:y_line, x] = 0
                elif self.edge_mode == 'lower_right':
                    mask_mat[y_line+1:h, x] = 0

        b_type = 'lower' if self.edge_mode == 'upper_left' else 'upper'
        mask_points, _, _ = extract_boundary_points(mask_mat, b_type)
        processed_region = np.zeros_like(mask_mat)

        guide_xs = np.where(np.any(guide_canvas > 0, axis=0))[0]
        mask_xs = mask_points[:, 0] if len(mask_points) else np.array([])
        all_x = np.unique(np.concatenate([mask_xs, guide_xs]))

        for x in all_x.astype(int):
            if x < 0 or x >= w:
                continue
            m_y_arr = mask_points[mask_points[:, 0] == x, 1] if len(mask_points) else np.array([])
            l_ys = np.where(guide_canvas[:, x] > 0)[0]

            if len(m_y_arr) > 0 and len(l_ys) > 0:
                y_mask, y_line = int(m_y_arr[0]), int(np.mean(l_ys))
                start, end = min(y_mask, y_line), max(y_mask, y_line)
                processed_region[start:end+1, x] = 255
            elif len(l_ys) > 0:
                y_line = int(np.mean(l_ys))
                if self.edge_mode == 'upper_left':
                    processed_region[y_line:h, x] = 255
                elif self.edge_mode == 'lower_right':
                    processed_region[0:y_line+1, x] = 255

        final_mask = cv2.bitwise_or(mask_mat, processed_region)
        self._save_layer_mask(layer, final_mask)


    def execute_undo_history_pop(self):
        """Pop the active layer's last undo state, write it to disk, and push the current state onto redo."""
        layer = self.active_layer()
        if layer is None or not layer.mask_files:
            return
        _, fn = self._find_file_by_frame(layer, self.current_idx)
        if not fn or not layer.undo_stacks.get(fn):
            return
        layer.redo_stacks.setdefault(fn, []).append(layer.current_mask_raw.copy())
        prev_binary = layer.undo_stacks[fn].pop()

        out_dir = layer.output_folder or layer.input_folder
        h, w = prev_binary.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        color_mask[prev_binary > 0] = [layer.color.blue(), layer.color.green(), layer.color.red()]

        cv2.imwrite(os.path.join(out_dir, fn), color_mask)
        layer.strokes.clear()

        layer.frame_guidelines.pop(self.current_idx, None)
        self.save_stroke_history_for_frame(layer)
        self.render_current_workspace_view()

    def execute_redo_history_pop(self):
        """Pop the active layer's last redo state, write it to disk, and push the current state onto undo."""
        layer = self.active_layer()
        if layer is None or not layer.mask_files:
            return
        _, fn = self._find_file_by_frame(layer, self.current_idx)
        if not fn or not layer.redo_stacks.get(fn):
            return
        layer.undo_stacks.setdefault(fn, []).append(layer.current_mask_raw.copy())
        fwd_binary = layer.redo_stacks[fn].pop()

        out_dir = layer.output_folder or layer.input_folder
        h, w = fwd_binary.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        color_mask[fwd_binary > 0] = [layer.color.blue(), layer.color.green(), layer.color.red()]

        cv2.imwrite(os.path.join(out_dir, fn), color_mask)
        layer.strokes.clear()
        self.save_stroke_history_for_frame(layer)
        self.render_current_workspace_view()


    def get_prev_line(self):
        """Restore the active layer's most recently drawn strokes from the buffer."""
        layer = self.active_layer()
        if layer is None:
            return
        if layer.last_drawn_strokes_buffer:
            layer.strokes = copy.deepcopy(layer.last_drawn_strokes_buffer)
            self.view.redraw_guidelines()

    def clear_drawing(self):
        """Clear the active layer's current strokes and saved guideline for this frame."""
        layer = self.active_layer()
        if layer:
            layer.strokes.clear()
            layer.frame_guidelines.pop(self.current_idx, None)
            self.save_stroke_history_for_frame(layer)
        self.view.redraw_guidelines()

    def change_edge_mode(self, index):
        """Set the active edge/clipping mode from a combo box selection.

        Args:
            index (int): Index of the selected item in the edge mode combo box.
        """
        modes = ["upper_left", "lower_right", "hole_fill", "hole_crop", "object"]
        self.edge_mode = modes[index]
        self.view.redraw_guidelines()
        for row in self._layer_row_widgets:
            row._sync_prop_param_stack()


    def _change_alpha(self, value):
        """Update the mask/video blend ratio from the alpha slider and re-render.

        Args:
            value (int): Slider value from 0 to 100 representing mask opacity percentage.
        """
        self.mask_alpha  = value / 100.0
        self.video_alpha = 1.0 - self.mask_alpha
        self.lbl_alpha_mask.setText(f"Mask\n{value}%")
        self.lbl_alpha_video.setText(f"Video\n{100 - value}%")
        self.render_current_workspace_view()

    def resizeEvent(self, event):
        """Refit the displayed image to the viewport whenever the main window is resized.

        Args:
            event (QResizeEvent): Resize event triggering this handler.
        """
        super().resizeEvent(event)
        if self.layers and not self.view.pixmap_item.pixmap().isNull():
            if hasattr(self, 'zoom_slider') and self.zoom_slider.value() != 100:
                self._update_zoom(self.zoom_slider.value())
            else:
                self.view.fitInView(self.view.pixmap_item, Qt.KeepAspectRatio)


    def trigger_external_propagation(self):
        """Start a background thread that propagates every layer's drawn strokes across future frames."""
        if getattr(self, '_prop_thread', None) and self._prop_thread.isRunning():
            self.statusBar.setText("System Status: Propagation already running…")
            return

        bg_source_type = getattr(self, 'bg_source_type', None)
        if bg_source_type == 'h5':
            if not self.h5_path or not os.path.exists(self.h5_path):
                self.statusBar.setText("System Status: No H5 video loaded — set a background source first.")
                return
            h5_path, h5_key = self.h5_path, self.h5_key
            video_source_factory = lambda: _h5_video_source(h5_path, h5_key)
        elif bg_source_type == 'folder':
            if not self.bg_folder_path or not os.path.isdir(self.bg_folder_path):
                self.statusBar.setText("System Status: No background image folder set.")
                return
            source = FolderVideoSource(self.bg_folder_path)
            video_source_factory = lambda: _static_video_source(source)
        elif bg_source_type == 'tif':
            if not self.bg_tif_path or not os.path.exists(self.bg_tif_path):
                self.statusBar.setText("System Status: No background TIFF loaded.")
                return
            if self.bg_tif_frames is None:
                try:
                    ret, frames = cv2.imreadmulti(self.bg_tif_path, flags=cv2.IMREAD_UNCHANGED)
                    self.bg_tif_frames = frames if ret and frames else []
                except Exception:
                    self.bg_tif_frames = []
            source = TifVideoSource(self.bg_tif_frames)
            video_source_factory = lambda: _static_video_source(source)
        else:
            self.statusBar.setText("System Status: No background source set — load a video, folder, or TIFF first.")
            return

        active_jobs = [
            (i, layer, [p for s in layer.strokes for p in s['pts']])
            for i, layer in enumerate(self.layers)
            if layer.strokes
        ]
        active_jobs = [(i, l, pts) for i, l, pts in active_jobs if pts]
        if not active_jobs:
            return

        steps       = self.spin_prop_steps.value()
        start       = self.current_idx
        frame_range = list(range(start, start + steps + 1))

        for _, layer, _ in active_jobs:
            self._snapshot_layer_frames_for_undo(layer, frame_range)

        from propagator.propagate_edge import _edge_setup, _edge_batch
        from propagator.propagate_poly import _poly_setup, _poly_batch
        from utils.shared_utils import make_batches
        batches = make_batches(frame_range[1:])

        layer_jobs = []
        for layer_idx, layer, flat_pts in active_jobs:
            ps   = layer.prop_settings
            mode = layer_propagation_mode(layer)
            c    = layer.color
            bgr  = [c.blue(), c.green(), c.red()]

            if mode in ('hole_fill', 'hole_crop', 'object'):
                layer_jobs.append(dict(
                    layer_idx=layer_idx,
                    setup_fn=_poly_setup, batch_fn=_poly_batch,
                    setup_kwargs=dict(
                        frame_range=frame_range, flat_pts=flat_pts, edge_mode=mode,
                        input_folder=layer.input_folder, output_folder=layer.output_folder,
                        target_color=bgr, n_samples=ps['samples'],
                        search_range=ps['search_range'],
                    ),
                ))
            else:
                layer_jobs.append(dict(
                    layer_idx=layer_idx,
                    setup_fn=_edge_setup, batch_fn=_edge_batch,
                    setup_kwargs=dict(
                        frame_range=frame_range, flat_pts=flat_pts, edge_mode=mode,
                        input_folder=layer.input_folder, output_folder=layer.output_folder,
                        target_color=bgr, n_samples=ps['samples'],
                        search_range=ps['search_range'],
                    ),
                ))

        self._prop_layers     = {i: layer for i, layer, _ in active_jobs}
        self._prop_start      = start
        self._prop_last_frame = start
        self._prop_errors     = []
        self._stop_event      = threading.Event()

        self._prop_thread = QThread(self)
        self._prop_worker = MultiLayerPropagationWorker(
            video_source_factory, batches, layer_jobs, self._stop_event)
        self._prop_worker.moveToThread(self._prop_thread)

        self._prop_thread.started.connect(self._prop_worker.run)
        self._prop_worker.batch_done.connect(self._on_prop_batch_done)
        self._prop_worker.layer_error.connect(self._on_prop_layer_error)
        self._prop_worker.finished.connect(self._on_prop_finished)
        self._prop_worker.finished.connect(self._prop_thread.quit)
        self._prop_worker.finished.connect(self._prop_worker.deleteLater)
        self._prop_thread.finished.connect(self._prop_thread.deleteLater)

        self.btn_prop.setEnabled(False)
        self.btn_stop_prop.setEnabled(True)
        n = len(layer_jobs)
        self.statusBar.setText(f"System Status: Propagating {n} layer(s)…")
        self._prop_thread.start()

    def stop_propagation(self):
        """Signal the running propagation worker to stop after its current batch."""
        stop_event = getattr(self, '_stop_event', None)
        if stop_event:
            stop_event.set()
        self.btn_stop_prop.setEnabled(False)
        self.statusBar.setText("System Status: Stopping after current batch…")

    def _on_prop_batch_done(self, layer_idx, batch_tl):
        """Store a completed propagation batch's results and jump the viewport to its first frame.

        Args:
            layer_idx (int): Index of the layer this batch belongs to.
            batch_tl (dict): Mapping of frame index to stroke list produced by the batch.
        """
        layers = getattr(self, '_prop_layers', {})
        layer  = layers.get(layer_idx)
        if not batch_tl or layer is None:
            return

        for f_idx, stroke_list in batch_tl.items():
            layer.frame_guidelines[f_idx] = stroke_list

        processed             = sorted(batch_tl.keys())
        self._prop_last_frame = max(getattr(self, '_prop_last_frame', self._prop_start),
                                    processed[-1])
        jump_frame       = processed[0]
        self.current_idx = jump_frame

        for li, l in layers.items():
            if jump_frame in l.frame_guidelines:
                l.strokes = copy.deepcopy(l.frame_guidelines[jump_frame])

        self.timeline_slider.blockSignals(True)
        self.spin_frame.blockSignals(True)
        self.timeline_slider.setValue(jump_frame)
        self.spin_frame.setValue(jump_frame)
        self.timeline_slider.blockSignals(False)
        self.spin_frame.blockSignals(False)

        self.render_current_workspace_view()
        self.statusBar.setText(
            f"System Status: Layer {layer_idx + 1} · frames {processed[0]}–{processed[-1]}"
        )

    def _on_prop_layer_error(self, layer_idx, message):
        """Record and surface a per-layer propagation failure instead of failing silently.

        Args:
            layer_idx (int): Index of the layer that failed.
            message (str): Description of the failure.
        """
        self._prop_errors = getattr(self, '_prop_errors', [])
        self._prop_errors.append((layer_idx, message))
        layers = getattr(self, '_prop_layers', {})
        layer  = layers.get(layer_idx)
        name   = layer.display_name(layer_idx) if layer is not None and hasattr(layer, 'display_name') else f"Layer {layer_idx + 1}"
        self.statusBar.setText(f"System Status: {name} — {message}")
        logger.warning("%s — %s", name, message)

    def _on_prop_finished(self, all_tl):
        """Merge the final propagation timeline into each layer and jump to the last processed frame.

        Args:
            all_tl (dict): Mapping of layer index to its full frame-to-stroke-list timeline.
        """
        layers = getattr(self, '_prop_layers', {})

        for layer_idx, full_tl in (all_tl or {}).items():
            layer = layers.get(layer_idx)
            if not full_tl or layer is None:
                continue
            for f_idx, stroke_list in full_tl.items():
                layer.frame_guidelines[f_idx] = stroke_list

        last = min(getattr(self, '_prop_last_frame', self._prop_start),
                   self.timeline_slider.maximum())

        for li, l in layers.items():
            l.strokes = copy.deepcopy(l.frame_guidelines.get(last, l.strokes))

        self.current_idx = last
        self.timeline_slider.blockSignals(True)
        self.spin_frame.blockSignals(True)
        self.timeline_slider.setValue(last)
        self.spin_frame.setValue(last)
        self.timeline_slider.blockSignals(False)
        self.spin_frame.blockSignals(False)

        self.render_current_workspace_view()

        stopped = getattr(self, '_stop_event', None) and self._stop_event.is_set()
        n_errors = len(getattr(self, '_prop_errors', []))
        if stopped:
            msg = "System Status: Propagation stopped."
        elif n_errors:
            msg = (f"System Status: Propagation complete with {n_errors} "
                   f"error(s) — see log for details.")
        else:
            msg = "System Status: Propagation complete."
        self.statusBar.setText(msg)

        self._prop_layers = None
        self._prop_thread = None
        self._prop_worker = None
        self._stop_event  = None
        self.btn_prop.setEnabled(True)
        self.btn_stop_prop.setEnabled(False)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    workspace = PyQtMaskEditorWorkspace()
    workspace.show()
    sys.exit(app.exec_())
