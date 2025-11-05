import cv2 as cv
import numpy as np


def _gaussian_pyramid(img: np.ndarray, levels: int):
    gp = [img]
    for _ in range(levels):
        img = cv.pyrDown(img)
        gp.append(img)
    return gp


def _laplacian_pyramid(img: np.ndarray, levels: int):
    gp = _gaussian_pyramid(img, levels)
    lp = [gp[-1]]
    for i in range(levels - 1, -1, -1):
        size = (gp[i].shape[1], gp[i].shape[0])
        GE = cv.pyrUp(gp[i + 1], dstsize=size)
        L = cv.subtract(gp[i], GE)
        lp.append(L)
    return lp


def pyramid_texture_blend(content_bgr: np.ndarray,
                          texture_bgr: np.ndarray,
                          levels: int = 4) -> np.ndarray:
    h, w = content_bgr.shape[:2]
    tex = cv.resize(texture_bgr, (w, h), interpolation=cv.INTER_CUBIC)

    # 使用自动生成的中间渐变掩膜来融合中心区域的纹理
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2.0, h / 2.0
    r = np.sqrt(((xx - cx) / (0.6 * w)) ** 2 + ((yy - cy) / (0.6 * h)) ** 2)
    mask = np.clip(1.0 - r, 0.0, 1.0)
    mask = cv.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=15, sigmaY=15)
    mask3 = cv.merge([mask, mask, mask])

    # 构建金字塔
    lp_content = _laplacian_pyramid(content_bgr.astype(np.float32), levels)
    lp_tex = _laplacian_pyramid(tex.astype(np.float32), levels)

    gp_mask = _gaussian_pyramid(mask3.astype(np.float32), levels)

    LS = []
    for lC, lT, m in zip(lp_content, lp_tex, gp_mask[::-1]):
        LS.append(lC * (1.0 - m) + lT * m)

    # 重建
    img = LS[0]
    for i in range(1, len(LS)):
        size = (LS[i].shape[1], LS[i].shape[0])
        img = cv.pyrUp(img, dstsize=size)
        img = cv.add(img, LS[i])

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

