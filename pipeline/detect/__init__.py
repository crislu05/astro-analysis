"""Source detection on the masked KPNO mosaic."""

from pipeline.detect.pipeline import DetectionResult, detect, detect_from_mask_state

__all__ = ["detect", "detect_from_mask_state", "DetectionResult"]