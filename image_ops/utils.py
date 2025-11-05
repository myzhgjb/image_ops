import cv2 as cv
import numpy as np
from PIL import Image
from typing import Tuple


def read_image(path: str) -> np.ndarray:
    img = cv.imdecode(np.fromfile(path, dtype=np.uint8), cv.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f'无法读取图像: {path}')
    return img


def save_image(path: str, image_bgr: np.ndarray) -> None:
    if image_bgr.dtype != np.uint8:
        image_bgr = ensure_uint8(image_bgr)
    ext = path.split('.')[-1]
    success, buf = cv.imencode('.' + ext, image_bgr)
    if not success:
        raise RuntimeError('图像编码失败')
    with open(path, 'wb') as f:
        buf.tofile(f)


def ensure_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def add_canvas_texture(image_bgr: np.ndarray, scale: float = 0.2, frequency: int = 8) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    y = np.linspace(0, np.pi * frequency, h, dtype=np.float32)
    x = np.linspace(0, np.pi * frequency, w, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing='ij')
    texture = (np.sin(xx) * np.cos(yy) + 1.0) * 0.5  # [0,1]
    texture = (texture * 255).astype(np.uint8)
    texture_bgr = cv.merge([texture, texture, texture])
    blended = cv.addWeighted(image_bgr, 1.0, texture_bgr, scale, 0.0)
    return blended


def resize_fit(img: np.ndarray, max_side: int = 1024) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return img
    nh, nw = int(h * scale), int(w * scale)
    return cv.resize(img, (nw, nh), interpolation=cv.INTER_AREA)


def alpha_blend(img_a: np.ndarray, img_b: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """简单的两图线性融合；自动缩放到相同尺寸。
    alpha 表示 A 的权重，范围 [0,1]。
    """
    alpha = float(max(0.0, min(1.0, alpha)))
    h, w = img_a.shape[:2]
    img_b = cv.resize(img_b, (w, h), interpolation=cv.INTER_LINEAR)
    out = cv.addWeighted(img_a.astype(np.float32), alpha, img_b.astype(np.float32), 1.0 - alpha, 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def to_pil(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB))


def from_pil(img_pil: Image.Image) -> np.ndarray:
    rgb = np.array(img_pil)
    return cv.cvtColor(rgb, cv.COLOR_RGB2BGR)


def make_edge_mask(gray: np.ndarray, low: int = 80, high: int = 160) -> np.ndarray:
    edges = cv.Canny(gray, low, high)
    edges = cv.dilate(edges, None)
    return edges

