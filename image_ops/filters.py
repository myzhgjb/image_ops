import cv2 as cv
import numpy as np
from typing import Literal
from .utils import add_canvas_texture, ensure_uint8, make_edge_mask


def sketch_effect(image_bgr: np.ndarray,
                  mode: Literal['gray', 'color'] = 'gray',
                  blur_ksize: int = 21,
                  sigma: float = 0,
                  edge_low: int = 60,
                  edge_high: int = 150) -> np.ndarray:
    gray = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    blur = cv.GaussianBlur(gray, (blur_ksize, blur_ksize), sigma)
    edges = make_edge_mask(blur, edge_low, edge_high)
    inv = 255 - edges
    if mode == 'gray':
        return cv.cvtColor(inv, cv.COLOR_GRAY2BGR)
    else:
        color = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
        color = cv.bilateralFilter(color, 9, 75, 75)
        return cv.bitwise_and(color, color, mask=inv)


def oil_painting_effect(image_bgr: np.ndarray,
                        method: Literal['bilateral', 'median'] = 'bilateral',
                        strength: int = 9,
                        quantization_levels: int = 24,
                        add_texture: bool = True) -> np.ndarray:
    if method == 'bilateral':
        smoothed = image_bgr.copy()
        for _ in range(max(1, strength // 3)):
            smoothed = cv.bilateralFilter(smoothed, d=9, sigmaColor=75, sigmaSpace=75)
    else:
        k = strength if strength % 2 == 1 else strength + 1
        smoothed = cv.medianBlur(image_bgr, k)

    # 颜色量化
    Z = smoothed.reshape((-1, 3))
    Z = np.float32(Z)
    K = max(2, quantization_levels // 3)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv.kmeans(Z, K, None, criteria, 1, cv.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    quantized = centers[labels.flatten()].reshape(smoothed.shape)

    if add_texture:
        quantized = add_canvas_texture(quantized, scale=0.15, frequency=10)
    return quantized


def cartoon_effect(image_bgr: np.ndarray,
                   bilateral_iterations: int = 6,
                   edge_low: int = 80,
                   edge_high: int = 160) -> np.ndarray:
    # 平滑颜色
    color = image_bgr.copy()
    for _ in range(max(1, bilateral_iterations)):
        color = cv.bilateralFilter(color, d=9, sigmaColor=75, sigmaSpace=75)

    # 边缘强化
    gray = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)
    edges = make_edge_mask(gray, edge_low, edge_high)
    edges_inv = cv.bitwise_not(edges)
    edges_inv = cv.cvtColor(edges_inv, cv.COLOR_GRAY2BGR)
    cartoon = cv.bitwise_and(color, edges_inv)
    return cartoon

