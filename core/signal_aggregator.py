"""
signal_aggregator.py

Combines expression + motion/posture cues over a short rolling window
(default 8 seconds) so the response engine reacts to a sustained
pattern rather than a single noisy frame. This matches the concept
paper's "Signal Aggregation Layer".
"""

import time
from collections import deque, Counter


class SignalAggregator:
    def __init__(self, window_seconds: float = 8.0):
        self.window_seconds = window_seconds
        # Each entry: (timestamp, expression_label, expression_confidence, head_down, still)
        self._readings = deque()

    def add_reading(self, expression_label, expression_confidence, head_down, still):
        now = time.time()
        self._readings.append((now, expression_label, expression_confidence, head_down, still))
        self._evict_old(now)

    def _evict_old(self, now):
        while self._readings and (now - self._readings[0][0]) > self.window_seconds:
            self._readings.popleft()

    def get_aggregate(self):
        """
        Returns a summary dict describing the dominant pattern over the
        current window:
        {
            "dominant_expression": str,
            "expression_ratio": float,   # fraction of window matching dominant_expression
            "avg_confidence": float,
            "head_down_ratio": float,    # fraction of readings with head_down True
            "still_ratio": float,        # fraction of readings with still True
            "sample_count": int,
        }
        Returns None if there are no readings yet.
        """
        if not self._readings:
            return None

        labels = [r[1] for r in self._readings]
        confidences = [r[2] for r in self._readings]
        head_down_flags = [r[3] for r in self._readings]
        still_flags = [r[4] for r in self._readings]

        label_counts = Counter(labels)
        dominant_label, dominant_count = label_counts.most_common(1)[0]

        return {
            "dominant_expression": dominant_label,
            "expression_ratio": dominant_count / len(labels),
            "avg_confidence": sum(confidences) / len(confidences),
            "head_down_ratio": sum(head_down_flags) / len(head_down_flags),
            "still_ratio": sum(still_flags) / len(still_flags),
            "sample_count": len(self._readings),
        }
