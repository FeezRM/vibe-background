"""Hard-cut detection, so per-shot state (matting memory, lighting averages) can be reset."""

import cv2
import numpy as np

# Hue/saturation histogram correlation below this between consecutive frames counts as a cut.
_CUT_CORRELATION = 0.6


class SceneCutDetector:
    def __init__(self):
        self._prev: np.ndarray | None = None

    def is_cut(self, frame: np.ndarray) -> bool:
        """frame: HxWx3 uint8 RGB. True if it starts a new shot (never for the first frame)."""
        small = cv2.resize(frame, (96, 54), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        prev, self._prev = self._prev, hist
        if prev is None:
            return False
        return cv2.compareHist(prev, hist, cv2.HISTCMP_CORREL) < _CUT_CORRELATION
