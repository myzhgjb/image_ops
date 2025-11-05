import numpy as np
import cv2 as cv


def _safe_import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torchvision import models
        return torch, nn, optim, models
    except Exception as e:
        raise ImportError('需要安装 PyTorch 与 torchvision 才能使用 neural 风格迁移') from e


def _to_tensor(image_bgr: np.ndarray, device):
    import torch
    image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(image_rgb.transpose(2, 0, 1)).unsqueeze(0)
    return t.to(device)


def _to_image(tensor) -> np.ndarray:
    image = tensor.detach().cpu().squeeze(0).clamp(0, 1).numpy().transpose(1, 2, 0)
    bgr = cv.cvtColor((image * 255.0).astype(np.uint8), cv.COLOR_RGB2BGR)
    return bgr


def _gram_matrix(x):
    import torch
    b, c, h, w = x.size()
    features = x.view(b, c, h * w)
    G = torch.bmm(features, features.transpose(1, 2))
    return G / (c * h * w)


def neural_style_transfer(content_bgr: np.ndarray,
                          style_bgr: np.ndarray,
                          steps: int = 300,
                          content_weight: float = 1.0,
                          style_weight: float = 5.0,
                          tv_weight: float = 1e-4,
                          max_side: int = 512) -> np.ndarray:
    """Gatys 风格迁移（优化生成图），默认 300 步可得到较强笔触效果。
    计算量随分辨率与步数上升，建议先将较长边缩放到 512。
    """
    torch, nn, optim, models = _safe_import_torch()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 尺寸对齐与下采样
    def resize_fit(img, max_side):
        h, w = img.shape[:2]
        s = max_side / max(h, w)
        if s < 1:
            img = cv.resize(img, (int(w * s), int(h * s)), interpolation=cv.INTER_AREA)
        return img

    content_bgr = resize_fit(content_bgr, max_side)
    style_bgr = cv.resize(style_bgr, (content_bgr.shape[1], content_bgr.shape[0]), interpolation=cv.INTER_CUBIC)

    content = _to_tensor(content_bgr, device)
    style = _to_tensor(style_bgr, device)
    generated = content.clone().requires_grad_(True)

    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
    for p in vgg.parameters():
        p.requires_grad_(False)

    # 选择若干层作为内容/风格层
    content_layers = ['21']  # relu4_2
    style_layers = ['0', '5', '10', '19', '28']  # relu1_1 ... relu5_1

    def extract_features(x):
        feats = {}
        for name, layer in vgg._modules.items():
            x = layer(x)
            if name in content_layers + style_layers:
                feats[name] = x
        return feats

    content_feats = extract_features(content)
    style_feats = extract_features(style)
    style_grams = {k: _gram_matrix(v) for k, v in style_feats.items() if k in style_layers}

    optimizer = optim.Adam([generated], lr=0.03)

    def tv_loss(img):
        x_diff = img[:, :, :, 1:] - img[:, :, :, :-1]
        y_diff = img[:, :, 1:, :] - img[:, :, :-1, :]
        return (x_diff.abs().mean() + y_diff.abs().mean())

    for i in range(steps):
        optimizer.zero_grad()
        gen_feats = extract_features(generated)

        c_loss = 0.0
        for cl in content_layers:
            c_loss = c_loss + (gen_feats[cl] - content_feats[cl]).pow(2).mean()

        s_loss = 0.0
        for sl in style_layers:
            Gg = _gram_matrix(gen_feats[sl])
            s_loss = s_loss + (Gg - style_grams[sl]).pow(2).mean()

        loss = content_weight * c_loss + style_weight * s_loss + tv_weight * tv_loss(generated)
        loss.backward()
        optimizer.step()

    return _to_image(generated)

