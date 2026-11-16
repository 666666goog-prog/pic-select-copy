"""离线 OCR 封装：RapidOCR（onnxruntime），返回 selection.OcrLine 列表。

RapidOCR 的原始结果是 [[四边形角点, 文本, 置信度], ...]，
这里统一转换成 selection.OcrLine，供 selection.build_document 切字。

召回策略（v1.0.1）：单次识别会漏掉小号低对比文字（X 暗色截图里的
灰色 @账号行最明显），漏掉的行在图上没有字框、选字时盖不上蓝，
所以改成多趟识别再合并去重：

- A 趟：主极性（深色底先反色成黑字白底，PP-OCR 对黑字白底最稳）；
- B 趟：反极性（只在整图模式跑，长图分片时跳过省时间）；
- C 趟：增强——A 趟图先拉对比度再放大 2 倍识别，坐标再除回 2，
  专门捞 A/B 漏掉的小字、灰字。

长图（高 > 2600）先按 1800 高、重叠 300 切成横片，每片跑 A、C 两趟，
坐标按片的 y 起点偏移后全局合并。极性（深/浅底）只按整图判定一次。
合并去重：候选行与已收行的框 IoU > 0.4，或任一方的中心落在对方框内，
就算重复，保留先收的那行（A 先于 B 先于 C）。
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from selection import OcrLine, Rect

TALL_THRESHOLD = 2600   # 图高超过这个值就分片识别
TILE_HEIGHT = 1800
TILE_OVERLAP = 300


def _invert(img: np.ndarray) -> np.ndarray:
    return (255 - img.astype(np.int32)).astype(np.uint8)


def _shift(lines, dy: float = 0.0, scale: float = 1.0):
    """把一批行的坐标整体缩放/平移（C 趟除 2、分片加 y 偏移用）。"""
    out = []
    for ln in lines:
        if isinstance(ln.points, Rect):
            r = ln.points
            pts = Rect(r.x0 * scale, r.y0 * scale + dy,
                       r.x1 * scale, r.y1 * scale + dy)
        else:
            pts = [(x * scale, y * scale + dy) for x, y in ln.points]
        out.append(OcrLine(text=ln.text, points=pts, score=ln.score))
    return out


def _iou(a: Rect, b: Rect) -> float:
    iw = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    ih = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
    inter = iw * ih
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


def _is_duplicate(cand: OcrLine, accepted) -> bool:
    cr = cand.rect
    for ln in accepted:
        r = ln.rect
        if _iou(cr, r) > 0.4:
            return True
        if r.contains(cr.cx, cr.cy) or cr.contains(r.cx, r.cy):
            return True
    return False


def _merge_into(accepted, candidates):
    for cand in candidates:
        if not _is_duplicate(cand, accepted):
            accepted.append(cand)
    return accepted


def _tiles(height: int):
    """长图分片：[(y0, y1), ...]，片高 1800、相邻重叠 300。"""
    step = TILE_HEIGHT - TILE_OVERLAP
    out = []
    y = 0
    while y < height:
        out.append((y, min(height, y + TILE_HEIGHT)))
        if y + TILE_HEIGHT >= height:
            break
        y += step
    return out


class OcrEngine:
    """懒加载 RapidOCR 引擎（模型随包内置，全程离线，多趟共用一个实例）。"""

    def __init__(self):
        self._engine = None

    def _ensure(self):
        if self._engine is None:
            from rapidocr_onnxruntime import RapidOCR
            self._engine = RapidOCR()
        return self._engine

    # ------------------------------------------------------------ 单趟
    def _run(self, img: np.ndarray):
        """对一幅图跑一次 RapidOCR，返回 list[OcrLine]（该图自身坐标）。"""
        engine = self._ensure()
        result, _elapse = engine(img)
        lines = []
        if not result:
            return lines
        for item in result:
            if not item or len(item) < 2:
                continue
            points, text = item[0], item[1]
            score = float(item[2]) if len(item) > 2 else 1.0
            text = (text or "").strip("\n")
            if not text.strip():
                continue
            pts = [(float(p[0]), float(p[1])) for p in points]
            lines.append(OcrLine(text=text, points=pts, score=score))
        return lines

    def _run_enhanced(self, a_img: np.ndarray):
        """C 趟：A 趟极性的图拉对比度 + 放大 2 倍识别，坐标除回 2。"""
        f = a_img.astype(np.float32)
        stretched = np.clip((f - 150.0) * 2.0 + 90.0, 0, 255).astype(np.uint8)
        big = Image.fromarray(stretched).resize(
            (stretched.shape[1] * 2, stretched.shape[0] * 2), Image.LANCZOS)
        return _shift(self._run(np.asarray(big)), scale=0.5)

    # ------------------------------------------------------------ 入口
    def recognize(self, image_rgb: np.ndarray):
        """image_rgb: HxWx3 的 RGB numpy 数组。返回 list[OcrLine]。"""
        img = np.ascontiguousarray(image_rgb)
        # 极性只按整图判定一次：整图平均亮度低 = 深色底
        gray_mean = float(img.astype(np.float32).mean(axis=2).mean())
        dark = gray_mean < 140
        height = img.shape[0]
        if height > TALL_THRESHOLD:
            return self._recognize_tiled(img, dark)
        return self._recognize_whole(img, dark)

    def _recognize_whole(self, img: np.ndarray, dark: bool):
        a_img = _invert(img) if dark else img
        accepted = self._run(a_img)                       # A：主极性
        b_img = img if dark else _invert(img)
        _merge_into(accepted, self._run(b_img))           # B：反极性
        _merge_into(accepted, self._run_enhanced(a_img))  # C：增强
        return accepted

    def _recognize_tiled(self, img: np.ndarray, dark: bool):
        accepted = []
        for y0, y1 in _tiles(img.shape[0]):
            tile = img[y0:y1]
            a_img = _invert(tile) if dark else tile
            _merge_into(accepted, _shift(self._run(a_img), dy=y0))
            _merge_into(accepted, _shift(self._run_enhanced(a_img), dy=y0))
        return accepted
