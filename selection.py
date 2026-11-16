"""选字核心逻辑（纯逻辑，不依赖 Qt，可独立单元测试）。

文档模型：一张图经 OCR 后得到若干文本行；本模块把每行拆成逐字框，
再按阅读顺序（先行带、后行内从左到右）排成一个扁平的字符列表。
选字模型和普通文本编辑器一致：用“光标位（caret）”表示字符之间的位置，
拖动 = 起点光标位到当前光标位之间的字符被选中。
复制文本 = 选中字符拼接，跨行时在行与行之间插入换行符。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------- 数据结构

@dataclass
class Rect:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1


@dataclass
class OcrLine:
    """OCR 引擎输出的一行：文本 + 四边形框（4 个角点）或轴对齐矩形。"""

    text: str
    points: list  # [(x, y), ...] 多边形角点；也可以直接给 Rect
    score: float = 1.0

    @property
    def rect(self) -> Rect:
        if isinstance(self.points, Rect):
            return self.points
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return Rect(min(xs), min(ys), max(xs), max(ys))


@dataclass
class CharBox:
    char: str
    rect: Rect
    line_id: int  # 阅读顺序下的行号（0 起）

    @property
    def is_space(self) -> bool:
        return self.char == " "


@dataclass
class LineSpan:
    line_id: int
    rect: Rect
    start: int  # 在扁平字符列表中的起始下标（包含）
    end: int    # 结束下标（不包含）


@dataclass
class Document:
    chars: list = field(default_factory=list)       # list[CharBox]，阅读顺序
    lines: list = field(default_factory=list)       # list[LineSpan]，阅读顺序
    stats: dict = field(default_factory=dict)       # {'projection': n, 'fallback': n}

    def __len__(self) -> int:
        return len(self.chars)

    @property
    def full_text(self) -> str:
        return self.text_for_range(0, len(self.chars))

    # ---------------------------------------------------------- 命中测试
    def char_at_point(self, x: float, y: float):
        """返回包含该点的字符下标；不在任何字符框内时返回 None。"""
        for i, c in enumerate(self.chars):
            if c.is_space:
                continue
            if c.rect.contains(x, y):
                return i
        return None

    def _line_span_for_y(self, y: float) -> LineSpan:
        """找 y 所在的行；不在任何行内时取竖直距离最近的行。"""
        best = None
        best_dist = None
        for span in self.lines:
            if span.rect.y0 <= y <= span.rect.y1:
                return span
            dist = span.rect.y0 - y if y < span.rect.y0 else y - span.rect.y1
            if best_dist is None or dist < best_dist:
                best, best_dist = span, dist
        return best

    def caret_at(self, x: float, y: float) -> int:
        """由图片坐标算出光标位（0..len(chars)）。

        规则：先定位到 y 所在（或最近）的行，再在该行内按 x 与每个字
        的中点比较：x 小于某字中点 -> 光标在这个字之前；扫完都没命中
        -> 光标在本行末尾。x 在行左外侧 -> 行首；右外侧 -> 行尾。
        这样在一个字的左半边按下、右半边松开，光标位恰好跨过这一个字，
        选中的就是这一个字（单个数字能被单独选中的关键）。
        """
        if not self.lines:
            return 0
        span = self._line_span_for_y(y)
        if span is None:
            return 0
        for i in range(span.start, span.end):
            c = self.chars[i]
            if x < c.rect.cx:
                return i
        return span.end

    # ---------------------------------------------------------- 选中文本
    def text_for_range(self, caret_a: int, caret_b: int) -> str:
        """两个光标位之间的文本；跨行插入换行符。"""
        lo, hi = sorted((caret_a, caret_b))
        lo = max(0, min(lo, len(self.chars)))
        hi = max(0, min(hi, len(self.chars)))
        out = []
        prev_line = None
        for i in range(lo, hi):
            c = self.chars[i]
            if prev_line is not None and c.line_id != prev_line:
                out.append("\n")
            out.append(c.char)
            prev_line = c.line_id
        return "".join(out)

    def selected_indices(self, caret_a: int, caret_b: int):
        lo, hi = sorted((caret_a, caret_b))
        lo = max(0, min(lo, len(self.chars)))
        hi = max(0, min(hi, len(self.chars)))
        return range(lo, hi)


# ---------------------------------------------------------- 阅读顺序

def order_lines(lines):
    """把 OCR 行按阅读顺序分组：竖直重叠的行归为同一行带（同一逻辑行），
    带按 y 排序、带内按 x 排序。返回 [ [OcrLine, ...], ... ]（行带列表）。

    OCR 有时会把同一行拆成左右两个框，它们是同一行而不是两行：
    复制时不应在中间插入换行，间距够大时补一个空格占位即可。
    """
    indexed = list(lines)
    indexed.sort(key=lambda ln: (ln.rect.cy, ln.rect.x0))
    bands = []  # 每带: [y0, y1, [lines...]]
    for ln in indexed:
        r = ln.rect
        placed = False
        for band in bands:
            overlap = min(r.y1, band[1]) - max(r.y0, band[0])
            smaller = min(r.height, band[1] - band[0])
            if smaller > 0 and overlap > 0.5 * smaller:
                band[2].append(ln)
                band[0] = min(band[0], r.y0)
                band[1] = max(band[1], r.y1)
                placed = True
                break
        if not placed:
            bands.append([r.y0, r.y1, [ln]])
    bands.sort(key=lambda b: b[0])
    result = []
    for _y0, _y1, members in bands:
        members.sort(key=lambda ln: ln.rect.x0)
        result.append(members)
    return result


# ---------------------------------------------------------- 逐字切分

def _dominant_bg(crop: np.ndarray) -> np.ndarray:
    """取裁剪区域的主导背景色：把颜色量化后出现最多的那一档的均值。"""
    flat = crop.reshape(-1, crop.shape[-1]).astype(np.int32)
    quant = (flat // 32)
    code = quant[:, 0] * 1024 + quant[:, 1] * 32 + quant[:, 2]
    vals, counts = np.unique(code, return_counts=True)
    main = vals[np.argmax(counts)]
    mask = code == main
    return flat[mask].mean(axis=0)


def _ink_runs(crop: np.ndarray):
    """竖直投影：返回墨迹列段 [(x_start, x_end), ...]（裁剪内相对坐标）。

    以与主导背景色的距离做二值化，深底白字和白底黑字都适用。
    """
    if crop.size == 0:
        return []
    arr = crop.astype(np.int32)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    bg = _dominant_bg(arr)
    dist = np.abs(arr - bg).sum(axis=2)
    ink = dist > 90  # 三通道距离和阈值，抗锯齿边缘也能检出
    col = ink.any(axis=0)
    runs = []
    start = None
    for x, v in enumerate(col):
        if v and start is None:
            start = x
        elif not v and start is not None:
            runs.append((start, x))
            start = None
    if start is not None:
        runs.append((start, len(col)))
    return runs


def _merge_runs_to_count(runs, target):
    """把列段按“最小间隙优先”逐对合并，直到段数等于 target。

    一个汉字/字母内部笔画断开会产生多个列段，字内间隙通常小于字间
    间隙，所以先合并最小间隙能把同一字的笔画段并回一个字。
    若列段数本来就少于 target（字与字粘连），无法靠合并解决，返回 None。
    """
    runs = list(runs)
    if len(runs) < target:
        return None
    while len(runs) > target:
        gaps = [runs[i + 1][0] - runs[i][1] for i in range(len(runs) - 1)]
        j = int(np.argmin(gaps))
        runs[j:j + 2] = [(runs[j][0], runs[j + 1][1])]
    return runs


def _line_char_boxes(text: str, rect: Rect, crop):
    """切一行，返回 (与 text 等长的 Rect 列表, 是否用了投影法)。

    空格没有墨迹：给它一个按平均字宽估算的窄框占位，只为保持下标对齐。
    """
    n = len(text)
    if n == 0:
        return [], True
    non_space_idx = [i for i, ch in enumerate(text) if ch != " "]

    used_projection = False
    segments = None
    if crop is not None and non_space_idx:
        runs = _ink_runs(crop)
        merged = _merge_runs_to_count(runs, len(non_space_idx)) if runs else None
        if merged is not None and len(merged) == len(non_space_idx):
            segments = merged
            used_projection = True

    boxes = [None] * n
    if used_projection:
        seg_iter = iter(segments)
        avg_w = rect.width / max(1, n)
        prev_end = rect.x0
        for i, ch in enumerate(text):
            if ch == " ":
                boxes[i] = Rect(prev_end, rect.y0, prev_end + avg_w, rect.y1)
            else:
                s, e = next(seg_iter)
                boxes[i] = Rect(rect.x0 + s, rect.y0, rect.x0 + e, rect.y1)
                prev_end = rect.x0 + e
    else:
        step = rect.width / n
        for i in range(n):
            boxes[i] = Rect(rect.x0 + i * step, rect.y0,
                            rect.x0 + (i + 1) * step, rect.y1)
    return boxes, used_projection


def build_document(lines, image=None) -> Document:
    """lines: list[OcrLine]；image: 整图 numpy 数组（RGB，用于投影切分，可为 None）。

    返回 Document，并统计每行用的是投影法还是等分回退（stats）。
    """
    doc = Document()
    bands = order_lines([ln for ln in lines if ln.text])
    proj = fall = 0
    for line_id, band in enumerate(bands):
        start = len(doc.chars)
        band_rect = None
        prev_seg_rect = None
        for ln in band:
            rect = ln.rect
            # 同一行带内左右两段之间空隙较大时，补一个空格占位（不换行）
            if prev_seg_rect is not None:
                gap = rect.x0 - prev_seg_rect.x1
                if gap > 4:
                    gy0 = min(prev_seg_rect.y0, rect.y0)
                    gy1 = max(prev_seg_rect.y1, rect.y1)
                    doc.chars.append(CharBox(
                        char=" ",
                        rect=Rect(prev_seg_rect.x1, gy0, rect.x0, gy1),
                        line_id=line_id))
            crop = None
            if image is not None:
                x0 = max(0, int(rect.x0)); y0 = max(0, int(rect.y0))
                x1 = min(image.shape[1], int(np.ceil(rect.x1)))
                y1 = min(image.shape[0], int(np.ceil(rect.y1)))
                if x1 > x0 and y1 > y0:
                    crop = image[y0:y1, x0:x1]
                    # 以裁剪的整图坐标为基准切分，列段相对坐标加上 x0 即整图坐标。
                    crop_rect = Rect(x0, y0, x1, y1)
                    boxes, used = _line_char_boxes(ln.text, crop_rect, crop)
                else:
                    boxes, used = _line_char_boxes(ln.text, rect, None)
            else:
                boxes, used = _line_char_boxes(ln.text, rect, None)
            if used:
                proj += 1
            else:
                fall += 1
            for ch, box in zip(ln.text, boxes):
                doc.chars.append(CharBox(char=ch, rect=box, line_id=line_id))
            prev_seg_rect = rect
            if band_rect is None:
                band_rect = Rect(rect.x0, rect.y0, rect.x1, rect.y1)
            else:
                band_rect = Rect(min(band_rect.x0, rect.x0),
                                 min(band_rect.y0, rect.y0),
                                 max(band_rect.x1, rect.x1),
                                 max(band_rect.y1, rect.y1))
        doc.lines.append(LineSpan(line_id=line_id, rect=band_rect,
                                  start=start, end=len(doc.chars)))
    doc.stats = {"projection": proj, "fallback": fall}
    return doc
