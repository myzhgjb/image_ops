import cv2 as cv
import numpy as np
from skimage.exposure import match_histograms


def _image_stats_lab(img_lab: np.ndarray):
    l, a, b = cv.split(img_lab)
    return (l.mean(), l.std() + 1e-6, a.mean(), a.std() + 1e-6, b.mean(), b.std() + 1e-6)


def color_transfer_lab(content_bgr: np.ndarray, style_bgr: np.ndarray) -> np.ndarray:
    content_lab = cv.cvtColor(content_bgr, cv.COLOR_BGR2LAB).astype(np.float32)
    style_lab = cv.cvtColor(style_bgr, cv.COLOR_BGR2LAB).astype(np.float32)

    lM, lS, aM, aS, bM, bS = _image_stats_lab(content_lab)
    lM2, lS2, aM2, aS2, bM2, bS2 = _image_stats_lab(style_lab)

    (lC, aC, bC) = cv.split(content_lab)
    lC = (lC - lM) * (lS2 / lS) + lM2
    aC = (aC - aM) * (aS2 / aS) + aM2
    bC = (bC - bM) * (bS2 / bS) + bM2

    transfer = cv.merge([lC, aC, bC])
    transfer = np.clip(transfer, 0, 255).astype(np.uint8)
    return cv.cvtColor(transfer, cv.COLOR_LAB2BGR)


def histogram_match_rgb(content_bgr: np.ndarray, style_bgr: np.ndarray) -> np.ndarray:
    content_rgb = cv.cvtColor(content_bgr, cv.COLOR_BGR2RGB)
    style_rgb = cv.cvtColor(style_bgr, cv.COLOR_BGR2RGB)
    matched = match_histograms(content_rgb, style_rgb, channel_axis=-1)
    matched = np.clip(matched, 0, 255).astype(np.uint8)
    return cv.cvtColor(matched, cv.COLOR_RGB2BGR)

