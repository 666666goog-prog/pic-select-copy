"""图片选字 v1.0.1 —— 把图片拖进来，直接在原图上选字复制。

只有两个功能：
1. 把一张图片拖进窗口打开（拖新图替换旧图）；
2. 识别完成后，在原图上用 I 光标拖着选字（最小一个字/一个数字），
   右键「复制」或 Ctrl+C 复制选中的字。
另有必要的看图支持：打开自动适窗；滚轮在图片上是以光标为中心缩放、
在图片外是上下滚动浏览长图；拖空白处平移。
"""

from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QThread, Signal
from PySide6.QtGui import (QAction, QClipboard, QColor, QCursor, QImage,
                           QKeySequence, QPainter, QPen, QPixmap, QShortcut)
from PySide6.QtWidgets import (QApplication, QMainWindow, QMenu, QWidget)

from ocr_engine import OcrEngine
from selection import Document, build_document

VERSION = "1.0.1"
APP_NAME = "图片选字"
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def drop_file_path(mime):
    """拖放验收规则（主窗口和图片区共用）：恰好拖入一个本地文件、
    且扩展名在允许列表里时，返回它的路径；否则返回 None。"""
    urls = mime.urls()
    if len(urls) == 1 and urls[0].isLocalFile():
        path = urls[0].toLocalFile()
        if os.path.splitext(path)[1].lower() in ALLOWED_EXTS:
            return path
    return None


# ---------------------------------------------------------------- OCR 线程

class OcrWorker(QObject):
    finished = Signal(object, int)   # Document, 代次
    failed = Signal(str, int)

    def __init__(self, image_rgb: np.ndarray, generation: int):
        super().__init__()
        self._image = image_rgb
        self._generation = generation
        self._engine = OcrEngine()

    def run(self):
        try:
            lines = self._engine.recognize(self._image)
            doc = build_document(lines, self._image)
            self.finished.emit(doc, self._generation)
        except Exception as exc:  # 把错误原文带回状态栏
            self.failed.emit(str(exc), self._generation)


# ---------------------------------------------------------------- 图片视图

class ImageView(QWidget):
    """负责画图、选字高亮、缩放平移、右键复制的控件。"""

    status_hint = Signal(str)   # 需要在状态栏提示时发给主窗口
    file_dropped = Signal(str)  # 图片拖到图片区上松手 -> 主窗口加载它

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        # 图片区盖满整个窗口，拖放必须由它自己接，否则图上拖图替换不了
        self.setAcceptDrops(True)
        self._pixmap: QPixmap | None = None
        self._doc: Document | None = None
        self._scale = 1.0
        self._offset = QPointF(0, 0)   # 图片左上角在控件中的位置
        self._user_transformed = False  # 用户手动缩放/平移过就不再自动适窗

        self._anchor = None   # 选字起点光标位
        self._caret = None    # 选字当前光标位
        self._selecting = False
        self._panning = False
        self._pan_last = None

    # ------------------------------------------------------------ 基础状态
    @property
    def document(self):
        return self._doc

    def set_image(self, pixmap: QPixmap):
        self._pixmap = pixmap
        self._doc = None
        self._anchor = self._caret = None
        self._user_transformed = False
        self._fit()
        self.update()

    def set_document(self, doc: Document):
        self._doc = doc
        self._anchor = self._caret = None
        self.update()

    def clear_selection(self):
        self._anchor = self._caret = None
        self.update()

    # ------------------------------------------------------------ 坐标变换
    def _fit(self):
        if self._pixmap is None or self.width() <= 0 or self.height() <= 0:
            return
        iw, ih = self._pixmap.width(), self._pixmap.height()
        self._scale = min(self.width() / iw, self.height() / ih)
        self._offset = QPointF((self.width() - iw * self._scale) / 2,
                               (self.height() - ih * self._scale) / 2)

    def _to_image(self, pos) -> QPointF:
        return QPointF((pos.x() - self._offset.x()) / self._scale,
                       (pos.y() - self._offset.y()) / self._scale)

    def _to_widget_rect(self, rect) -> QRectF:
        return QRectF(rect.x0 * self._scale + self._offset.x(),
                      rect.y0 * self._scale + self._offset.y(),
                      rect.width * self._scale,
                      rect.height * self._scale)

    # ------------------------------------------------------------ 选字
    def set_selection(self, anchor: int, caret: int):
        """测试与内部共用：直接设置选区（两个光标位）。"""
        self._anchor, self._caret = anchor, caret
        self.update()

    def selection_range(self):
        if self._anchor is None or self._caret is None:
            return None
        if self._anchor == self._caret:
            return None
        return (min(self._anchor, self._caret), max(self._anchor, self._caret))

    def selected_text(self) -> str:
        rng = self.selection_range()
        if rng is None or self._doc is None:
            return ""
        return self._doc.text_for_range(rng[0], rng[1])

    def copy_selection(self):
        text = self.selected_text()
        if text:
            QApplication.clipboard().setText(text)
            self.status_hint.emit(f"已复制 {len(text.replace(chr(10), ''))} 个字")
        return text

    # ------------------------------------------------------------ 绘制
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(45, 45, 45))
        if self._pixmap is None:
            painter.setPen(QColor(200, 200, 200))
            font = painter.font()
            font.setPointSize(16)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignCenter, "把图片拖进来")
            return
        target = QRectF(self._offset.x(), self._offset.y(),
                        self._pixmap.width() * self._scale,
                        self._pixmap.height() * self._scale)
        painter.drawPixmap(target, self._pixmap, QRectF(self._pixmap.rect()))
        rng = self.selection_range()
        if rng is not None and self._doc is not None:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(51, 153, 255, 110))
            for i in self._doc.selected_indices(rng[0], rng[1]):
                c = self._doc.chars[i]
                if c.is_space:
                    continue
                painter.drawRect(self._to_widget_rect(c.rect))
        if self._selecting and self._caret is not None and self._doc is not None:
            # 拖动中在当前光标位画一条细竖线，提示选到哪了
            pos = self._caret_pos_point()
            if pos is not None:
                painter.setPen(QPen(QColor(51, 153, 255), 1))
                painter.drawLine(pos[0], pos[1])

    def _caret_pos_point(self):
        """当前光标位对应的控件坐标竖线（取相邻字的边缘）。"""
        doc = self._doc
        caret = self._caret
        if caret is None or len(doc) == 0:
            return None
        if caret < len(doc):
            r = self._to_widget_rect(doc.chars[caret].rect)
            return (QPointF(r.left(), r.top()), QPointF(r.left(), r.bottom()))
        r = self._to_widget_rect(doc.chars[-1].rect)
        return (QPointF(r.right(), r.top()), QPointF(r.right(), r.bottom()))

    # ------------------------------------------------------------ 鼠标
    def mousePressEvent(self, event):
        if self._pixmap is None:
            return
        if event.button() == Qt.MiddleButton or (
                event.button() == Qt.LeftButton
                and event.modifiers() & Qt.AltModifier):
            self._start_pan(event.position())
            event.accept()
            return
        if event.button() == Qt.LeftButton:
            img = self._to_image(event.position())
            hit = (self._doc.char_at_point(img.x(), img.y())
                   if self._doc is not None else None)
            if hit is not None:
                caret = self._doc.caret_at(img.x(), img.y())
                self._anchor = caret
                self._caret = caret
                self._selecting = True
                self.update()
            else:
                self._start_pan(event.position())
            event.accept()

    def _start_pan(self, pos):
        self._panning = True
        self._pan_last = pos
        self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self._selecting and self._doc is not None:
            img = self._to_image(event.position())
            self._caret = self._doc.caret_at(img.x(), img.y())
            self.update()
            return
        if self._panning and self._pan_last is not None:
            delta = event.position() - self._pan_last
            self._pan_last = event.position()
            self._offset += delta
            self._user_transformed = True
            self.update()
            return
        # 悬停：文字上 I 光标，其它地方箭头
        if self._pixmap is not None and self._doc is not None:
            img = self._to_image(event.position())
            if self._doc.char_at_point(img.x(), img.y()) is not None:
                self.setCursor(Qt.IBeamCursor)
            else:
                self.setCursor(Qt.ArrowCursor)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._selecting:
            self._selecting = False
            self.update()
        if self._panning and event.button() in (Qt.LeftButton, Qt.MiddleButton):
            self._panning = False
            self._pan_last = None
            self.setCursor(Qt.ArrowCursor)

    # ------------------------------------------------------------ 拖放
    def dragEnterEvent(self, event):
        if drop_file_path(event.mimeData()) is not None:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if drop_file_path(event.mimeData()) is not None:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        path = drop_file_path(event.mimeData())
        if path is not None:
            self.file_dropped.emit(path)
            event.acceptProposedAction()
        else:
            event.ignore()

    # ------------------------------------------------------------ 滚轮
    def handle_wheel(self, pos: QPointF, delta_y: int):
        """滚轮行为（独立成方法方便测试）：
        光标在图片上 -> 以光标为中心缩放；
        光标在图片外 -> 上下滚动浏览（长图），图完全装得下时不滚、保持居中。"""
        if self._pixmap is None or delta_y == 0:
            return
        disp = QRectF(self._offset.x(), self._offset.y(),
                      self._pixmap.width() * self._scale,
                      self._pixmap.height() * self._scale)
        if disp.contains(pos):
            factor = 1.25 if delta_y > 0 else 0.8
            new_scale = max(0.02, min(30.0, self._scale * factor))
            if new_scale == self._scale:
                return
            img_before = self._to_image(pos)
            self._scale = new_scale
            # 让光标下的图片点保持不动
            self._offset = QPointF(pos.x() - img_before.x() * self._scale,
                                   pos.y() - img_before.y() * self._scale)
            self._user_transformed = True
            self.update()
            return
        disp_h = disp.height()
        if disp_h <= self.height():
            return  # 整个图都看得见，没什么可滚的
        # 每格滚轮（120）滚 80px：上滚图往下走（回到上面），下滚图往上走
        new_y = self._offset.y() + (delta_y / 120.0) * 80.0
        new_y = max(self.height() - disp_h, min(0.0, new_y))
        if new_y != self._offset.y():
            self._offset = QPointF(self._offset.x(), new_y)
            self._user_transformed = True
            self.update()

    def wheelEvent(self, event):
        if self._pixmap is None:
            return
        self.handle_wheel(event.position(), event.angleDelta().y())
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._pixmap is not None and not self._user_transformed:
            self._fit()

    # ------------------------------------------------------------ 右键菜单
    def contextMenuEvent(self, event):
        if self._doc is None:
            return
        menu = QMenu(self)
        act = QAction("复制", self)
        act.setEnabled(bool(self.selected_text()))
        act.triggered.connect(self.copy_selection)
        menu.addAction(act)
        menu.exec(event.globalPos())


# ---------------------------------------------------------------- 主窗口

class MainWindow(QMainWindow):
    ocr_done = Signal()    # 供测试等待识别完成

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{VERSION}")
        self.resize(960, 720)
        self.setAcceptDrops(True)

        self.view = ImageView(self)
        self.setCentralWidget(self.view)
        self.view.status_hint.connect(self.statusBar().showMessage)
        self.view.file_dropped.connect(self.load_image)
        self.statusBar().showMessage("把图片拖进来")

        copy_shortcut = QShortcut(QKeySequence.Copy, self)
        copy_shortcut.activated.connect(self.view.copy_selection)

        self._thread: QThread | None = None
        self._worker: OcrWorker | None = None
        self._retiring = []   # 换图时退下来的旧识别线程（还在跑完手头那张）
        self._pending_path = None
        self._ocr_gen = 0

    # ------------------------------------------------------------ 拖放
    # 窗口边框等图片区没盖住的地方拖入时走这里（兜底）；图片区自己也接。
    def dragEnterEvent(self, event):
        if drop_file_path(event.mimeData()) is not None:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        path = drop_file_path(event.mimeData())
        if path is not None:
            self.load_image(path)
            event.acceptProposedAction()

    # ------------------------------------------------------------ 加载+识别
    def load_image(self, path: str):
        """拖放和测试共用的入口：打开图片并自动开始识别。"""
        try:
            pil = Image.open(path).convert("RGB")
        except Exception as exc:
            self.statusBar().showMessage(f"打开图片失败：{exc}")
            return
        qimg = QImage(pil.tobytes(), pil.width, pil.height,
                      pil.width * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg.copy())
        self.view.set_image(pixmap)
        self.statusBar().showMessage("正在识别…")
        self._pending_path = path
        self._start_ocr(np.asarray(pil, dtype=np.uint8))

    def _start_ocr(self, image_rgb: np.ndarray):
        self._retire_thread()
        self._ocr_gen += 1
        self._thread = QThread(self)
        self._worker = OcrWorker(image_rgb, self._ocr_gen)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_ocr_finished)
        self._worker.failed.connect(self._on_ocr_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.start()

    def _retire_thread(self):
        """把上一轮识别线程退下来：不在界面线程上等它（旧实现 wait(3000)
        会把换图卡住 3 秒），让它自己跑完手头那张图自然结束；
        它的结果回来时靠代次守卫丢弃。已结束的旧条目顺手清掉。"""
        self._retiring = [e for e in self._retiring if e[0].isRunning()]
        thread, worker = self._thread, self._worker
        self._thread = None
        self._worker = None
        if thread is not None and thread.isRunning():
            self._retiring.append((thread, worker))
            thread.quit()

    def _on_ocr_finished(self, doc: Document, generation: int):
        if generation != self._ocr_gen:
            return  # 旧图的结果作废，已经换新图了
        self.view.set_document(doc)
        if len(doc) == 0:
            self.statusBar().showMessage("识别完成：没有认出文字")
        else:
            self.statusBar().showMessage("识别完成，在图片上拖着选字，右键复制")
        self.ocr_done.emit()

    def _on_ocr_failed(self, message: str, generation: int):
        if generation != self._ocr_gen:
            return
        self.statusBar().showMessage(f"识别失败：{message}")
        self.ocr_done.emit()

    def closeEvent(self, event):
        threads = [t for t, _w in self._retiring]
        if self._thread is not None:
            threads.append(self._thread)
        for t in threads:
            t.quit()
        for t in threads:
            t.wait(1000)   # 关窗口时最多短等一下，不长卡
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        win.load_image(sys.argv[1])
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
