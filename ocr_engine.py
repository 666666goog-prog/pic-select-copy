"""离线 OCR 封装：RapidOCR（onnxruntime），返回 selection.OcrLine 列表。

RapidOCR 的原始结果是 [[四边形角点, 文本, 置信度], ...]，
这里统一转换成 selection.OcrLine，供 selection.build_document 切字。
"""

from __future__ import annotations

import numpy as np

from selection import OcrLine


class OcrEngine:
    """懒加载 RapidOCR 引擎（模型随包内置，全程离线）。"""

    def __init__(self):
        self._engine = None

    def _ensure(self):
        if self._engine is None:
            from rapidocr_onnxruntime import RapidOCR
            self._engine = RapidOCR()
        return self._engine

    def recognize(self, image_rgb: np.ndarray):
        """image_rgb: HxWx3 的 RGB numpy 数组。返回 list[OcrLine]。

        深色底白字（如 X 的暗色模式截图）先反色再识别：PP-OCR 对
        黑字白底最稳，反色不改变坐标，认出的框可以直接用回原图。
        整图平均亮度低判定为深色底。
        """
        engine = self._ensure()
        img = image_rgb
        gray = img.astype(np.float32).mean(axis=2)
        if float(gray.mean()) < 140:
            img = (255 - img.astype(np.int32)).astype(np.uint8)
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
