import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QRect, QPoint, Signal
from PySide6.QtGui import (
    QImage, QPixmap, QPainter, QPen, QColor, QFont, QIcon, QFontMetrics,
    QDragEnterEvent, QDropEvent, QMouseEvent, QPaintEvent, QKeyEvent,
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QListWidget, QListWidgetItem,
    QFileDialog, QMessageBox, QGroupBox, QSlider, QSplitter,
    QProgressDialog, QStatusBar, QFrame,
)

from core.image_io import read_image, write_image
from core.redactor import redact_image


def resource_path(filename: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return base / filename


def app_icon() -> QIcon:
    return QIcon(str(resource_path("icon.ico")))


# ---------------------------------------------------------------------------
# ImageCanvas: interactive drawing, moving, resizing
# ---------------------------------------------------------------------------

class ImageCanvas(QWidget):
    region_added = Signal(QRect)
    region_duplicated = Signal(QRect)
    region_selected = Signal(int)
    region_moved = Signal(int, tuple)
    region_resized = Signal(int, tuple)

    HANDLE_SIZE = 8
    HANDLES = {'LT': (-1, -1), 'RT': (1, -1), 'LB': (-1, 1), 'RB': (1, 1)}

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image: np.ndarray | None = None
        self._display_pixmap: QPixmap | None = None
        self._scale = 1.0
        self._offset = QPoint(0, 0)
        self._regions: list[dict] = []

        self._mode = "view"
        self._selected_idx = -1
        self._hover_handle: str | None = None
        self._hover_idx = -1

        # Drag / resize / move state
        self._action: str | None = None
        self._action_idx = -1
        self._action_handle: str | None = None
        self._start_rect: tuple | None = None
        self._start_pos = QPoint()
        self._drag_current = QPoint()

        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

    # -- public API --

    def set_image(self, image: np.ndarray):
        self._image = image.copy() if image is not None else None
        self._regions = []
        self._selected_idx = -1
        self._update_display()

    def set_regions(self, regions: list[tuple | dict]):
        self._regions = []
        for region in regions:
            if isinstance(region, dict):
                self._regions.append({
                    'rect': tuple(region['rect']),
                    'enabled': bool(region.get('enabled', True)),
                })
            else:
                self._regions.append({'rect': tuple(region), 'enabled': True})
        self._selected_idx = -1
        self.update()

    def get_regions(self) -> list[tuple]:
        return [r['rect'] for r in self._regions]

    def get_region_states(self) -> list[dict]:
        return [
            {'rect': tuple(r['rect']), 'enabled': bool(r.get('enabled', True))}
            for r in self._regions
        ]

    def get_enabled_regions(self) -> list[tuple]:
        return [r['rect'] for r in self._regions if r['enabled']]

    def clear_regions(self):
        self._regions = []
        self._selected_idx = -1
        self.update()

    def delete_region(self, idx: int):
        if 0 <= idx < len(self._regions):
            del self._regions[idx]
            if self._selected_idx == idx:
                self._selected_idx = -1
            elif self._selected_idx > idx:
                self._selected_idx -= 1
            self.update()

    def set_region_enabled(self, idx: int, enabled: bool):
        if 0 <= idx < len(self._regions):
            self._regions[idx]['enabled'] = enabled
            self.update()

    def set_mode(self, mode: str):
        self._mode = mode
        self._action = None
        self.update()

    def select_region(self, idx: int):
        self._selected_idx = idx if 0 <= idx < len(self._regions) else -1
        self.update()

    # -- coordinate helpers --

    def _img_to_widget(self, x: int, y: int) -> QPoint:
        return QPoint(int(x * self._scale + self._offset.x()),
                      int(y * self._scale + self._offset.y()))

    def _widget_to_img(self, wx: int, wy: int) -> tuple:
        x = int((wx - self._offset.x()) / self._scale)
        y = int((wy - self._offset.y()) / self._scale)
        return x, y

    # -- hit testing --

    def _handle_of_region(self, idx: int, wx: int, wy: int) -> str | None:
        x, y, w, h = self._regions[idx]['rect']
        p1 = self._img_to_widget(x, y)
        p2 = self._img_to_widget(x + w, y + h)
        pts = {
            'LT': (p1.x(), p1.y()),
            'RT': (p2.x(), p1.y()),
            'LB': (p1.x(), p2.y()),
            'RB': (p2.x(), p2.y()),
        }
        for name, (hx, hy) in pts.items():
            if abs(wx - hx) <= self.HANDLE_SIZE and abs(wy - hy) <= self.HANDLE_SIZE:
                return name
        return None

    def _handle_at(self, wx: int, wy: int) -> tuple:
        if self._selected_idx < 0:
            return -1, None
        handle = self._handle_of_region(self._selected_idx, wx, wy)
        if handle:
            return self._selected_idx, handle
        return -1, None

    def _region_at(self, wx: int, wy: int) -> int:
        ix, iy = self._widget_to_img(wx, wy)
        for i in range(len(self._regions) - 1, -1, -1):
            x, y, w, h = self._regions[i]['rect']
            if x <= ix <= x + w and y <= iy <= y + h:
                return i
        return -1

    # -- paint --

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(17, 24, 39))
        if not self._display_pixmap:
            return

        painter.drawPixmap(self._offset, self._display_pixmap)

        for i, region in enumerate(self._regions):
            x, y, w, h = region['rect']
            p1 = self._img_to_widget(x, y)
            p2 = self._img_to_widget(x + w, y + h)
            rect = QRect(p1, p2)

            is_selected = (i == self._selected_idx)
            color = QColor(239, 68, 68) if region['enabled'] else QColor(148, 163, 184)
            if is_selected:
                color = QColor(37, 99, 235)

            pen = QPen(color, 2 if is_selected else 2,
                       Qt.SolidLine if region['enabled'] else Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(rect)

            font = QFont("Microsoft YaHei", 9)
            painter.setFont(font)
            label = f"区域 {i + 1}"
            if not region['enabled']:
                label += " (已禁用)"
            painter.setPen(QPen(color))
            painter.drawText(p1.x(), p1.y() - 4, label)

            # Draw handles if selected
            if is_selected:
                hs = self.HANDLE_SIZE
                painter.setBrush(QColor(37, 99, 235))
                painter.setPen(QPen(QColor(255, 255, 255), 1))
                for pt in [(p1.x(), p1.y()), (p2.x(), p1.y()),
                           (p1.x(), p2.y()), (p2.x(), p2.y())]:
                    painter.drawRect(pt[0] - hs, pt[1] - hs, hs * 2, hs * 2)
                painter.setBrush(Qt.NoBrush)

        # Draw drag rect for new region
        if self._mode == "draw" and self._action == "draw":
            r = QRect(self._start_pos, self._drag_current).normalized()
            painter.setPen(QPen(QColor(16, 185, 129), 2, Qt.DashLine))
            painter.drawRect(r)

    # -- mouse events --

    def mousePressEvent(self, event: QMouseEvent):
        if self._image is None or event.button() != Qt.LeftButton:
            return

        pos = event.pos()

        if self._mode == "draw":
            self._action = "draw"
            self._start_pos = pos
            self._drag_current = pos
            self.update()
            return

        # view mode - check handles for ALL regions (topmost first)
        for i in range(len(self._regions) - 1, -1, -1):
            handle = self._handle_of_region(i, pos.x(), pos.y())
            if handle:
                self._selected_idx = i
                self._action = "resize"
                self._action_idx = i
                self._action_handle = handle
                self._start_rect = self._regions[i]['rect']
                self._start_pos = pos
                self.region_selected.emit(i)
                self.update()
                return

        # Check region interior
        idx = self._region_at(pos.x(), pos.y())
        if idx >= 0:
            self._selected_idx = idx
            self._action = "move"
            self._action_idx = idx
            self._start_rect = self._regions[idx]['rect']
            self._start_pos = pos
            self.region_selected.emit(idx)
        else:
            self._selected_idx = -1
            self.region_selected.emit(-1)
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        pos = event.pos()

        if self._mode == "draw" and self._action == "draw":
            self._drag_current = pos
            self.update()
            return

        if self._action == "resize" and self._start_rect:
            nx, ny, nw, nh = self._calc_resize(self._start_rect, self._action_handle,
                                               self._start_pos, pos)
            self._regions[self._action_idx]['rect'] = (nx, ny, nw, nh)
            self.update()
            return

        if self._action == "move" and self._start_rect:
            nx, ny, nw, nh = self._calc_move(self._start_rect, self._start_pos, pos)
            self._regions[self._action_idx]['rect'] = (nx, ny, nw, nh)
            self.update()
            return

        # Hover detection
        idx, handle = self._handle_at(pos.x(), pos.y())
        if handle:
            self._hover_handle = handle
            self.setCursor(self._cursor_for_handle(handle))
            return

        idx = self._region_at(pos.x(), pos.y())
        if idx >= 0:
            self._hover_idx = idx
            self.setCursor(Qt.OpenHandCursor)
        else:
            self._hover_idx = -1
            self._hover_handle = None
            self.setCursor(Qt.ArrowCursor)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() != Qt.LeftButton:
            return

        if self._mode == "draw" and self._action == "draw":
            r = QRect(self._start_pos, self._drag_current).normalized()
            if r.width() > 10 and r.height() > 10:
                x1, y1 = self._widget_to_img(r.left(), r.top())
                x2, y2 = self._widget_to_img(r.right(), r.bottom())
                self.region_added.emit(QRect(x1, y1, x2 - x1, y2 - y1))
            self._action = None
            self.update()
            return

        if self._action == "resize":
            new_rect = self._regions[self._action_idx]['rect']
            if new_rect != self._start_rect:
                self.region_resized.emit(self._action_idx, new_rect)
            self._action = None
            self._start_rect = None
            return

        if self._action == "move":
            new_rect = self._regions[self._action_idx]['rect']
            if new_rect != self._start_rect:
                self.region_moved.emit(self._action_idx, new_rect)
            self._action = None
            self._start_rect = None
            return

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if self._image is None or event.button() != Qt.LeftButton:
            return
        pos = event.pos()
        idx = self._region_at(pos.x(), pos.y())
        if idx >= 0:
            x, y, w, h = self._regions[idx]['rect']
            # Duplicate to the right with a 20px offset
            nx = x + w + 20
            ny = y
            # Clamp to image bounds
            img_h, img_w = self._image.shape[:2]
            if nx + w > img_w:
                nx = x - w - 20
                if nx < 0:
                    nx = x
                    ny = y + h + 20
                    if ny + h > img_h:
                        ny = y - h - 20
                        if ny < 0:
                            ny = y
            self.region_duplicated.emit(QRect(nx, ny, w, h))

    def _cursor_for_handle(self, handle: str):
        if handle in ('LT', 'RB'):
            return Qt.SizeFDiagCursor
        if handle in ('RT', 'LB'):
            return Qt.SizeBDiagCursor
        return Qt.ArrowCursor

    def _calc_resize(self, orig: tuple, handle: str, start: QPoint, curr: QPoint) -> tuple:
        sx, sy, sw, sh = orig
        dx = int((curr.x() - start.x()) / self._scale)
        dy = int((curr.y() - start.y()) / self._scale)
        min_size = 10

        left = sx
        top = sy
        right = sx + sw
        bottom = sy + sh

        if 'L' in handle:
            left = min(sx + dx, right - min_size)
        if 'R' in handle:
            right = max(right + dx, left + min_size)
        if 'T' in handle:
            top = min(sy + dy, bottom - min_size)
        if 'B' in handle:
            bottom = max(bottom + dy, top + min_size)

        if self._image is not None:
            img_h, img_w = self._image.shape[:2]
            left = max(0, min(left, img_w - min_size))
            top = max(0, min(top, img_h - min_size))
            right = max(left + min_size, min(right, img_w))
            bottom = max(top + min_size, min(bottom, img_h))

        return left, top, right - left, bottom - top

    def _calc_move(self, orig: tuple, start: QPoint, curr: QPoint) -> tuple:
        sx, sy, sw, sh = orig
        dx = int((curr.x() - start.x()) / self._scale)
        dy = int((curr.y() - start.y()) / self._scale)
        return sx + dx, sy + dy, sw, sh

    def resizeEvent(self, event):
        self._update_display()

    def _update_display(self):
        if self._image is None:
            self._display_pixmap = None
            self.update()
            return
        rgb = cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._display_pixmap = scaled
        self._scale = scaled.width() / pixmap.width()
        self._offset = QPoint((self.width() - scaled.width()) // 2,
                              (self.height() - scaled.height()) // 2)
        self.update()


# ---------------------------------------------------------------------------
# ModeSelector: fixed below-popup mode picker
# ---------------------------------------------------------------------------

class ModeSelector(QWidget):
    currentIndexChanged = Signal(int)
    H_PADDING = 20
    ARROW_WIDTH = 16

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items: list[str] = []
        self._current_index = -1

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._button = QPushButton()
        self._button.setObjectName("modeSelectorButton")
        self._button.clicked.connect(self._show_popup)
        layout.addWidget(self._button)

        self._popup = QFrame(None, Qt.Popup | Qt.FramelessWindowHint)
        self._popup.setObjectName("modeSelectorPopup")
        self._popup_layout = QVBoxLayout(self._popup)
        self._popup_layout.setContentsMargins(0, 0, 0, 0)
        self._popup_layout.setSpacing(1)

    def addItems(self, items: list[str]):
        self._items = list(items)
        width = self._calc_width()
        self.setFixedWidth(width)
        self._button.setFixedSize(width, 32)
        for i, text in enumerate(self._items):
            btn = QPushButton(text)
            btn.setObjectName("modeSelectorItem")
            btn.setFixedSize(width, 30)
            btn.clicked.connect(lambda checked=False, idx=i: self._select_index(idx))
            self._popup_layout.addWidget(btn)
        if self._items:
            self._select_index(0, emit=False)

    def currentText(self) -> str:
        if 0 <= self._current_index < len(self._items):
            return self._items[self._current_index]
        return ""

    def setCurrentText(self, text: str):
        if text in self._items:
            self._select_index(self._items.index(text))

    def _show_popup(self):
        self._popup.adjustSize()
        pos = self.mapToGlobal(QPoint(0, self.height() + 2))
        self._popup.move(pos)
        self._popup.show()

    def _select_index(self, index: int, emit: bool = True):
        if not 0 <= index < len(self._items):
            return
        self._current_index = index
        self._button.setText(f"{self._items[index]} ▼")
        self._popup.hide()
        if emit:
            self.currentIndexChanged.emit(index)

    def _calc_width(self) -> int:
        metrics = QFontMetrics(self.font())
        text_width = max((metrics.horizontalAdvance(text) for text in self._items), default=72)
        return text_width + self.H_PADDING + self.ARROW_WIDTH


# ---------------------------------------------------------------------------
# MainWindow
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("轻量化打码工具")
        self.setWindowIcon(app_icon())
        self.setMinimumSize(1400, 900)

        self._source_image: np.ndarray | None = None
        self._current_result: np.ndarray | None = None

        # Batch processing
        self._batch_paths: list[str] = []
        self._batch_idx = -1
        self._batch_regions: dict[int, list[dict]] = {}

        # Undo history stores both rectangle geometry and enabled state.
        self._history: list[list[dict]] = []
        self._history_idx = -1
        self._max_history = 50

        self._setup_ui()
        self._setup_theme()
        self._setup_shortcuts()

    # -- UI setup --

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        central.setObjectName("centralWidget")
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(12, 12, 12, 12)

        # ===== Left: image canvases =====
        left_panel = QVBoxLayout()
        left_panel.setSpacing(10)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)
        toolbar.setContentsMargins(0, 0, 0, 0)

        self.btn_load = QPushButton("打开图片")
        self.btn_load.setProperty("variant", "primary")
        self.btn_load.clicked.connect(self._load_image)
        toolbar.addWidget(self.btn_load)

        self.btn_load_folder = QPushButton("打开文件夹")
        self.btn_load_folder.setProperty("variant", "secondary")
        self.btn_load_folder.setToolTip("批量打开文件夹中的图片")
        self.btn_load_folder.setMinimumWidth(90)
        self.btn_load_folder.clicked.connect(self._load_folder)
        toolbar.addWidget(self.btn_load_folder)

        toolbar.addSpacing(10)

        self.btn_draw = QPushButton("框选模式")
        self.btn_draw.setProperty("variant", "accent")
        self.btn_draw.setCheckable(True)
        self.btn_draw.toggled.connect(self._toggle_draw_mode)
        self.btn_draw.setEnabled(False)
        toolbar.addWidget(self.btn_draw)

        # Square preset buttons with icons
        self.btn_square_small = QPushButton("小")
        self.btn_square_small.setProperty("variant", "compact")
        self.btn_square_small.setIcon(self._create_square_icon(8))
        self.btn_square_small.setToolTip("小正方形 (100x100)")
        self.btn_square_small.clicked.connect(lambda: self._add_preset("square", 100))
        self.btn_square_small.setEnabled(False)
        toolbar.addWidget(self.btn_square_small)

        self.btn_square_medium = QPushButton("中")
        self.btn_square_medium.setProperty("variant", "compact")
        self.btn_square_medium.setIcon(self._create_square_icon(14))
        self.btn_square_medium.setToolTip("中正方形 (170x170)")
        self.btn_square_medium.clicked.connect(lambda: self._add_preset("square", 170))
        self.btn_square_medium.setEnabled(False)
        toolbar.addWidget(self.btn_square_medium)

        self.btn_square_large = QPushButton("大")
        self.btn_square_large.setProperty("variant", "compact")
        self.btn_square_large.setIcon(self._create_square_icon(20))
        self.btn_square_large.setToolTip("大正方形 (240x240)")
        self.btn_square_large.clicked.connect(lambda: self._add_preset("square", 240))
        self.btn_square_large.setEnabled(False)
        toolbar.addWidget(self.btn_square_large)

        toolbar.addSpacing(10)

        self.btn_undo = QPushButton("撤销")
        self.btn_undo.setProperty("variant", "secondary")
        self.btn_undo.clicked.connect(self._undo)
        self.btn_undo.setEnabled(False)
        toolbar.addWidget(self.btn_undo)

        self.btn_clear_all = QPushButton("清空所有")
        self.btn_clear_all.setProperty("variant", "warning")
        self.btn_clear_all.clicked.connect(self._clear_all_regions)
        self.btn_clear_all.setEnabled(False)
        toolbar.addWidget(self.btn_clear_all)

        self.btn_delete = QPushButton("删除")
        self.btn_delete.setProperty("variant", "danger")
        self.btn_delete.clicked.connect(self._delete_selected)
        self.btn_delete.setEnabled(False)
        toolbar.addWidget(self.btn_delete)

        toolbar.addStretch()

        toolbar.addWidget(QLabel("打码模式:"))
        self.combo_mode = ModeSelector()
        self.combo_mode.addItems(["马赛克", "高斯模糊", "纯色块"])
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        toolbar.addWidget(self.combo_mode)

        # Color picker for solid mode
        self._solid_color_rgb = (0, 0, 0)  # Default black (RGB)
        self.color_picker_widget = QWidget()
        self.color_picker_widget.setObjectName("colorPicker")
        self.color_picker_widget.setFixedSize(98, 30)
        color_lo = QHBoxLayout(self.color_picker_widget)
        color_lo.setContentsMargins(0, 0, 0, 0)
        color_lo.setSpacing(4)
        color_lo.setAlignment(Qt.AlignVCenter)
        self._color_buttons = []
        colors = [
            ((0, 0, 0), "黑"),
            ((128, 128, 128), "灰"),
            ((255, 255, 255), "白"),
        ]
        for rgb, name in colors:
            btn = QPushButton()
            btn.setFixedSize(22, 22)
            btn.setProperty("variant", "color")
            btn.setToolTip(name)
            btn.clicked.connect(lambda checked=False, c=rgb: self._set_solid_color(c))
            self._color_buttons.append(btn)
            color_lo.addWidget(btn)
        self._refresh_color_buttons()
        self._set_color_picker_visible(False)
        toolbar.addWidget(self.color_picker_widget)

        self.btn_export = QPushButton("导出当前")
        self.btn_export.setProperty("variant", "primary")
        self.btn_export.clicked.connect(self._export_image)
        self.btn_export.setEnabled(False)
        toolbar.addWidget(self.btn_export)

        self.btn_batch_export = QPushButton("批量导出")
        self.btn_batch_export.setProperty("variant", "primary")
        self.btn_batch_export.clicked.connect(self._batch_export)
        self.btn_batch_export.setEnabled(False)
        toolbar.addWidget(self.btn_batch_export)

        left_panel.addLayout(toolbar)

        # Splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.setObjectName("imageSplitter")

        src_grp = QGroupBox("原图")
        src_lo = QVBoxLayout(src_grp)
        src_lo.setContentsMargins(10, 14, 10, 10)
        self.canvas_source = ImageCanvas()
        self.canvas_source.region_added.connect(self._on_region_added)
        self.canvas_source.region_duplicated.connect(self._on_region_duplicated)
        self.canvas_source.region_selected.connect(self._on_region_selected)
        self.canvas_source.region_moved.connect(self._on_region_moved)
        self.canvas_source.region_resized.connect(self._on_region_resized)
        src_lo.addWidget(self.canvas_source)
        splitter.addWidget(src_grp)

        pre_grp = QGroupBox("预览")
        pre_lo = QVBoxLayout(pre_grp)
        pre_lo.setContentsMargins(10, 14, 10, 10)
        self.canvas_preview = ImageCanvas()
        self.canvas_preview.set_mode("view")
        pre_lo.addWidget(self.canvas_preview)
        splitter.addWidget(pre_grp)

        splitter.setSizes([650, 650])
        left_panel.addWidget(splitter, 1)
        main_layout.addLayout(left_panel, 3)

        # ===== Right: region list + batch list =====
        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)

        # Region list
        reg_grp = QGroupBox("打码区域")
        reg_lo = QVBoxLayout(reg_grp)
        reg_lo.setContentsMargins(10, 14, 10, 10)
        reg_lo.setSpacing(10)

        self.list_regions = QListWidget()
        self.list_regions.setAlternatingRowColors(True)
        self.list_regions.itemChanged.connect(self._on_list_item_changed)
        self.list_regions.currentRowChanged.connect(self._on_list_row_changed)
        reg_lo.addWidget(self.list_regions)

        intensity_lo = QHBoxLayout()
        intensity_lo.addWidget(QLabel("打码强度:"))
        self.slider_intensity = QSlider(Qt.Horizontal)
        self.slider_intensity.setMinimumHeight(28)
        self.slider_intensity.setRange(3, 40)
        self.slider_intensity.setValue(12)
        self.slider_intensity.valueChanged.connect(self._update_preview)
        intensity_lo.addWidget(self.slider_intensity)
        reg_lo.addLayout(intensity_lo)

        right_panel.addWidget(reg_grp)

        # Batch list
        batch_grp = QGroupBox("批量图片")
        batch_lo = QVBoxLayout(batch_grp)
        batch_lo.setContentsMargins(10, 14, 10, 10)
        self.list_batch = QListWidget()
        self.list_batch.setAlternatingRowColors(True)
        self.list_batch.itemClicked.connect(self._on_batch_item_clicked)
        batch_lo.addWidget(self.list_batch)
        right_panel.addWidget(batch_grp)

        tips = QLabel(
            "快捷键:\n"
            "Ctrl+Z 撤销 | Ctrl+Y 清空所有\n"
            "Delete 删除选中 | Esc 取消框选/取消选中\n"
            "G 切换框选模式 | 双击区域复制选框"
        )
        tips.setObjectName("tipsLabel")
        right_panel.addWidget(tips)

        main_layout.addLayout(right_panel, 1)

        self.setAcceptDrops(True)
        self.statusBar().showMessage("就绪")

    def _setup_theme(self):
        self.setStyleSheet("""
            QMainWindow {
                background: #f5f7fb;
                color: #172033;
                font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
                font-size: 13px;
            }
            QWidget#centralWidget {
                background: #f5f7fb;
            }
            QGroupBox {
                background: #ffffff;
                border: 1px solid #dfe5ef;
                border-radius: 8px;
                margin-top: 10px;
                font-weight: 600;
                color: #27364f;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 12px;
                padding: 0 5px;
                background: #ffffff;
            }
            QPushButton {
                min-height: 30px;
                padding: 5px 12px;
                border-radius: 6px;
                border: 1px solid #cfd7e6;
                background: #ffffff;
                color: #243047;
                font-weight: 500;
            }
            QPushButton:hover {
                background: #f3f7ff;
                border-color: #9db7e8;
            }
            QPushButton:pressed {
                background: #e8f0ff;
            }
            QPushButton:disabled {
                background: #eef1f6;
                border-color: #d9dee8;
                color: #9aa5b6;
            }
            QPushButton[variant="primary"] {
                background: #2563eb;
                border-color: #2563eb;
                color: #ffffff;
            }
            QPushButton[variant="primary"]:hover {
                background: #1d4ed8;
                border-color: #1d4ed8;
            }
            QPushButton[variant="accent"] {
                background: #ecfdf5;
                border-color: #91d8bd;
                color: #047857;
            }
            QPushButton[variant="accent"]:hover {
                background: #d8f8e9;
                border-color: #34d399;
                color: #047857;
            }
            QPushButton[variant="accent"]:checked {
                background: #10b981;
                border-color: #10b981;
                color: #ffffff;
            }
            QPushButton[variant="accent"]:checked:hover {
                background: #059669;
                border-color: #059669;
                color: #ffffff;
            }
            QPushButton[variant="compact"] {
                min-width: 38px;
                padding: 4px 8px;
                background: #f8fafc;
            }
            QPushButton[variant="color"] {
                min-width: 22px;
                min-height: 22px;
                max-width: 22px;
                max-height: 22px;
                padding: 0;
                border-radius: 11px;
            }
            QPushButton[variant="warning"] {
                color: #9a5b00;
                border-color: #f0c878;
                background: #fff8e8;
            }
            QPushButton[variant="warning"]:hover {
                color: #8a4d00;
                border-color: #e7b84f;
                background: #fff1c2;
            }
            QPushButton[variant="danger"] {
                color: #b42318;
                border-color: #f2b8b5;
                background: #fff1f1;
            }
            QPushButton[variant="danger"]:hover {
                color: #991b1b;
                border-color: #ef8f8a;
                background: #ffe4e4;
            }
            QPushButton#modeSelectorButton {
                min-height: 32px;
                max-height: 32px;
                padding: 4px 8px;
                border: 1px solid #b8c4d6;
                border-radius: 6px;
                background: #ffffff;
                color: #243047;
                text-align: left;
            }
            QPushButton#modeSelectorButton:hover {
                border-color: #7ea0d8;
                background: #ffffff;
            }
            QPushButton#modeSelectorButton:focus {
                border: 1px solid #2563eb;
            }
            QFrame#modeSelectorPopup {
                background: transparent;
                border: 0;
            }
            QPushButton#modeSelectorItem {
                min-height: 30px;
                max-height: 30px;
                padding: 4px 10px;
                border: 0;
                border-radius: 6px;
                background: #ffffff;
                color: #243047;
                text-align: left;
            }
            QPushButton#modeSelectorItem:hover {
                background: #e8f0ff;
                color: #1d4ed8;
            }
            QWidget#colorPicker {
                background: transparent;
            }
            QWidget#colorPicker:disabled {
                opacity: 0.45;
            }
            QListWidget {
                background: #ffffff;
                border: 1px solid #e0e6f0;
                border-radius: 6px;
                padding: 4px;
                outline: 0;
                alternate-background-color: #f8fafc;
            }
            QListWidget::item {
                min-height: 28px;
                padding: 4px 6px;
                border-radius: 4px;
            }
            QListWidget::item:selected {
                background: #e8f0ff;
                color: #1d4ed8;
            }
            QSlider::groove:horizontal {
                height: 6px;
                border-radius: 3px;
                background: #d9e1ee;
            }
            QSlider::sub-page:horizontal {
                border-radius: 3px;
                background: #2563eb;
            }
            QSlider::handle:horizontal {
                width: 18px;
                height: 18px;
                margin: -7px 0;
                border-radius: 9px;
                background: #ffffff;
                border: 2px solid #2563eb;
            }
            QSplitter::handle {
                background: #e5ebf5;
            }
            QLabel#tipsLabel {
                padding: 10px 12px;
                border: 1px solid #dfe7f3;
                border-radius: 8px;
                background: #f8fbff;
                color: #64748b;
                font-size: 12px;
                line-height: 1.4;
            }
            QStatusBar {
                background: #ffffff;
                border-top: 1px solid #dfe5ef;
                color: #64748b;
            }
            ImageCanvas {
                background: #111827;
                border-radius: 6px;
            }
        """)

    # -- shortcuts --

    def _setup_shortcuts(self):
        # Let the canvas receive key events
        self.setFocusPolicy(Qt.StrongFocus)

    def _create_square_icon(self, size_px: int) -> QIcon:
        px = QPixmap(24, 24)
        px.fill(Qt.transparent)
        painter = QPainter(px)
        painter.setRenderHint(QPainter.Antialiasing)
        offset = (24 - size_px) // 2
        rect = QRect(offset, offset, size_px, size_px)
        painter.setPen(QPen(QColor(71, 85, 105), 2))
        painter.setBrush(QColor(241, 245, 249))
        painter.drawRect(rect)
        painter.end()
        return QIcon(px)

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        mods = event.modifiers()

        if key == Qt.Key_Escape:
            if self.btn_draw.isChecked():
                self.btn_draw.setChecked(False)
            else:
                self.canvas_source.select_region(-1)
                self.btn_delete.setEnabled(False)
            return

        if key == Qt.Key_G:
            self.btn_draw.setChecked(not self.btn_draw.isChecked())
            return

        if mods & Qt.ControlModifier:
            if key == Qt.Key_Z:
                self._undo()
                return
            if key == Qt.Key_Y:
                self._clear_all_regions()
                return

        if key == Qt.Key_Delete:
            self._delete_selected()
            return

        super().keyPressEvent(event)

    # -- drag & drop --

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDragEnterEvent):
        urls = event.mimeData().urls()
        if not urls:
            return
        paths = [u.toLocalFile() for u in urls]
        images = [p for p in paths if p.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]
        if images:
            if len(images) == 1:
                self._open_image_path(images[0])
            else:
                self._open_batch(images)

    # -- image loading --

    def _load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if path:
            self._open_image_path(path)

    def _load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if not folder:
            return
        exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
        paths = sorted([str(p) for p in Path(folder).iterdir()
                        if p.suffix.lower() in exts])
        if not paths:
            QMessageBox.information(self, "提示", "该文件夹中没有图片")
            return
        self._open_batch(paths)

    def _open_image_path(self, path: str):
        self._batch_paths = [path]
        self._batch_idx = 0
        self._batch_regions.clear()
        self.list_batch.clear()
        self.list_batch.addItem(Path(path).name)
        self._load_current_image()

    def _open_batch(self, paths: list[str]):
        self._batch_paths = paths
        self._batch_idx = 0
        self._batch_regions.clear()
        self.list_batch.clear()
        for p in paths:
            self.list_batch.addItem(Path(p).name)
        self._load_current_image()

    def _load_current_image(self):
        if not self._batch_paths or self._batch_idx < 0:
            return
        path = self._batch_paths[self._batch_idx]
        image = read_image(path)
        if image is None:
            QMessageBox.critical(self, "错误", f"无法加载图片:\n{path}")
            return

        self._source_image = image
        self.canvas_source.set_image(image)
        self.canvas_preview.set_image(image)
        self._current_result = image.copy()

        # Restore regions for this batch image if available
        saved = self._batch_regions.get(self._batch_idx, [])
        if saved:
            self.canvas_source.set_regions(list(saved))
            self._history = [list(saved)]
        else:
            self._history = [[]]
        self._history_idx = 0

        # Enable buttons
        self.btn_draw.setEnabled(True)
        self.btn_square_small.setEnabled(True)
        self.btn_square_medium.setEnabled(True)
        self.btn_square_large.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.btn_batch_export.setEnabled(len(self._batch_paths) > 0)

        self._sync_region_list()
        self._update_preview()
        self._sync_history_buttons()

        fname = Path(path).name
        self.statusBar().showMessage(f"已加载: {fname} ({self._batch_idx + 1}/{len(self._batch_paths)})")

    def _on_batch_item_clicked(self, item: QListWidgetItem):
        row = self.list_batch.row(item)
        if row == self._batch_idx:
            return
        # Save current regions before switching
        if 0 <= self._batch_idx < len(self._batch_paths):
            self._batch_regions[self._batch_idx] = self.canvas_source.get_region_states()
        self._batch_idx = row
        self._load_current_image()

    # -- region management --

    def _push_history(self):
        regions = self.canvas_source.get_region_states()
        # Remove redo branch
        self._history = self._history[:self._history_idx + 1]
        self._history.append(list(regions))
        if len(self._history) > self._max_history:
            self._history.pop(0)
        else:
            self._history_idx += 1
        self._sync_history_buttons()

    def _sync_history_buttons(self):
        self.btn_undo.setEnabled(self._history_idx > 0)
        has_regions = len(self.canvas_source.get_regions()) > 0
        self.btn_clear_all.setEnabled(has_regions)

    def _undo(self):
        if self._history_idx <= 0:
            return
        self._history_idx -= 1
        regions = list(self._history[self._history_idx])
        self.canvas_source.set_regions(regions)
        self._sync_region_list()
        self._update_preview()
        self._sync_history_buttons()

    def _clear_all_regions(self):
        if not self.canvas_source.get_regions():
            return
        self.canvas_source.clear_regions()
        self._sync_region_list()
        self._update_preview()
        self._push_history()
        self.btn_delete.setEnabled(False)

    def _sync_region_list(self):
        self.list_regions.blockSignals(True)
        self.list_regions.clear()
        for i, r in enumerate(self.canvas_source.get_regions()):
            item = QListWidgetItem(f"区域 {i + 1}  ({r[2]}x{r[3]})")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            enabled = self.canvas_source._regions[i].get('enabled', True)
            item.setCheckState(Qt.Checked if enabled else Qt.Unchecked)
            item.setData(Qt.UserRole, i)
            self.list_regions.addItem(item)
        self.list_regions.blockSignals(False)
        # Restore selection without triggering unwanted side effects
        idx = self.canvas_source._selected_idx
        if idx >= 0:
            self.list_regions.setCurrentRow(idx)
            self.btn_delete.setEnabled(True)
        else:
            self.btn_delete.setEnabled(False)

    def _toggle_draw_mode(self, checked: bool):
        self.canvas_source.set_mode("draw" if checked else "view")
        self.btn_draw.setText("退出框选" if checked else "框选模式")
        self.statusBar().showMessage("框选模式：拖拽画框，可连续画多个，按 Esc 退出" if checked else "浏览模式")

    def _on_region_added(self, qrect: QRect):
        rect = (qrect.x(), qrect.y(), qrect.width(), qrect.height())
        self.canvas_source._regions.append({'rect': rect, 'enabled': True})
        self.canvas_source.select_region(len(self.canvas_source._regions) - 1)
        self._sync_region_list()
        self._update_preview()
        self._push_history()
        # Stay in draw mode for continuous drawing

    def _on_region_duplicated(self, qrect: QRect):
        rect = (qrect.x(), qrect.y(), qrect.width(), qrect.height())
        self.canvas_source._regions.append({'rect': rect, 'enabled': True})
        self.canvas_source.select_region(len(self.canvas_source._regions) - 1)
        self._sync_region_list()
        self._update_preview()
        self._push_history()
        # Exit draw mode if active
        if self.btn_draw.isChecked():
            self.btn_draw.setChecked(False)

    def _on_region_selected(self, idx: int):
        self.list_regions.setCurrentRow(idx)
        self.btn_delete.setEnabled(idx >= 0)

    def _on_region_moved(self, idx: int, rect: tuple):
        self._sync_region_list()
        self._update_preview()
        self._push_history()

    def _on_region_resized(self, idx: int, rect: tuple):
        self._sync_region_list()
        self._update_preview()
        self._push_history()

    def _on_list_item_changed(self, item: QListWidgetItem):
        idx = item.data(Qt.UserRole)
        enabled = item.checkState() == Qt.Checked
        self.canvas_source.set_region_enabled(idx, enabled)
        self._update_preview()
        self._push_history()

    def _on_list_row_changed(self, row: int):
        self.canvas_source.select_region(row)
        self.btn_delete.setEnabled(row >= 0)

    def _delete_selected(self):
        idx = self.canvas_source._selected_idx
        if idx < 0:
            return
        self.canvas_source.delete_region(idx)
        self._sync_region_list()
        self._update_preview()
        self._push_history()
        self.btn_delete.setEnabled(False)

    def _add_preset(self, shape: str, size: int = 70):
        if self._source_image is None:
            return
        h, w = self._source_image.shape[:2]
        cx, cy = w // 4, h // 3  # Default position on left side

        if shape == "square":
            rect = (cx - size // 2, cy - size // 2, size, size)
        else:
            rect = (cx - 40, cy - 25, 80, 50)

        self.canvas_source._regions.append({'rect': rect, 'enabled': True})
        self.canvas_source.select_region(len(self.canvas_source._regions) - 1)
        self._sync_region_list()
        self._update_preview()
        self._push_history()

    def _clear_regions(self):
        self.canvas_source.clear_regions()
        self.list_regions.clear()
        self._update_preview()
        self._push_history()

    def _on_mode_changed(self, index: int):
        is_solid = self.combo_mode.currentText() == "纯色块"
        self._set_color_picker_visible(is_solid)
        self._update_preview()

    def _set_solid_color(self, rgb: tuple):
        self._solid_color_rgb = rgb
        self._refresh_color_buttons()
        self._update_preview()

    def _refresh_color_buttons(self):
        for btn, rgb in zip(self._color_buttons, [(0, 0, 0), (128, 128, 128), (255, 255, 255)]):
            r, g, b = rgb
            selected = rgb == self._solid_color_rgb
            border = "#1677ff" if selected else "#999"
            width = 3 if selected else 2
            hover_border = "#1677ff" if selected else "#333"
            hover_width = 3 if selected else 2
            btn.setStyleSheet(
                f"QPushButton {{ background-color: rgb({r},{g},{b}); border: {width}px solid {border}; border-radius: 11px; }}"
                f"QPushButton:hover {{ border: {hover_width}px solid {hover_border}; }}"
            )

    def _set_color_picker_visible(self, visible: bool):
        for btn in self._color_buttons:
            btn.setVisible(visible)

    # -- preview & export --

    def _update_preview(self):
        if self._source_image is None:
            return
        regions = self.canvas_source.get_enabled_regions()
        if not regions:
            self._current_result = self._source_image.copy()
            self.canvas_preview.set_image(self._source_image)
            return

        mode_map = {"马赛克": "mosaic", "高斯模糊": "blur", "纯色块": "solid"}
        mode = mode_map.get(self.combo_mode.currentText(), "mosaic")
        intensity = self.slider_intensity.value()

        if mode == "mosaic":
            result = redact_image(self._source_image, regions, mode, block_size=intensity)
        elif mode == "blur":
            result = redact_image(self._source_image, regions, mode, ksize=intensity * 2 + 1)
        else:
            r, g, b = self._solid_color_rgb
            color_bgr = (b, g, r)
            result = redact_image(self._source_image, regions, mode, color=color_bgr)

        self._current_result = result
        self.canvas_preview.set_image(result)

    def _export_image(self):
        if self._current_result is None:
            return
        if not self._batch_paths:
            return
        src = Path(self._batch_paths[self._batch_idx])
        default_name = src.stem + "_redacted" + src.suffix
        path, _ = QFileDialog.getSaveFileName(
            self, "保存图片", str(default_name),
            "PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp)")
        if path:
            if write_image(path, self._current_result):
                self.statusBar().showMessage(f"已保存: {path}")
            else:
                QMessageBox.critical(self, "错误", f"无法保存图片:\n{path}")

    def _batch_export(self):
        if not self._batch_paths:
            return

        # Save current image's regions first
        if 0 <= self._batch_idx < len(self._batch_paths):
            self._batch_regions[self._batch_idx] = self.canvas_source.get_region_states()

        # Check if any image has enabled regions
        has_any = any(
            any(region.get('enabled', True) for region in self._batch_regions.get(i, []))
            for i in range(len(self._batch_paths))
        )
        if not has_any:
            QMessageBox.warning(self, "提示", "没有框选区域，请先框选")
            return

        folder = QFileDialog.getExistingDirectory(self, "选择导出文件夹")
        if not folder:
            return

        progress = QProgressDialog("批量处理中...", "取消", 0, len(self._batch_paths), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        mode_map = {"马赛克": "mosaic", "高斯模糊": "blur", "纯色块": "solid"}
        mode = mode_map.get(self.combo_mode.currentText(), "mosaic")
        intensity = self.slider_intensity.value()

        saved = 0
        for i, path in enumerate(self._batch_paths):
            progress.setValue(i)
            QApplication.processEvents()
            if progress.wasCanceled():
                break

            img = read_image(path)
            if img is None:
                continue

            regions = [
                tuple(region['rect'])
                for region in self._batch_regions.get(i, [])
                if region.get('enabled', True)
            ]
            if regions:
                if mode == "mosaic":
                    result = redact_image(img, regions, mode, block_size=intensity)
                elif mode == "blur":
                    result = redact_image(img, regions, mode, ksize=intensity * 2 + 1)
                else:
                    r, g, b = self._solid_color_rgb
                    color_bgr = (b, g, r)
                    result = redact_image(img, regions, mode, color=color_bgr)
            else:
                result = img

            src = Path(path)
            out_path = Path(folder) / (src.stem + "_redacted" + src.suffix)
            if write_image(out_path, result):
                saved += 1

        progress.setValue(len(self._batch_paths))
        QMessageBox.information(self, "完成", f"成功处理 {saved}/{len(self._batch_paths)} 张图片")
        self.statusBar().showMessage(f"批量导出完成: {saved} 张")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setWindowIcon(app_icon())
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
