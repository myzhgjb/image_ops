import argparse
from image_ops import (
    read_image, save_image,
    sketch_effect, oil_painting_effect, cartoon_effect,
    color_transfer_lab, histogram_match_rgb,
    pyramid_texture_blend, alpha_blend, neural_style_transfer
)


def run_single(content_path: str, style_path: str, method: str, out_path: str,
               alpha: float = 0.5, levels: int = 4,
               steps: int = 300, content_weight: float = 1.0, style_weight: float = 5.0):
    content = read_image(content_path)
    style = read_image(style_path) if style_path else None

    if method == 'sketch':
        res = sketch_effect(content, 'gray', 21, 0, 60, 150)
    elif method == 'oil':
        res = oil_painting_effect(content, 'bilateral', 9, 24, True)
    elif method == 'cartoon':
        res = cartoon_effect(content, 6, 80, 160)
    elif method == 'color':
        if style is None:
            raise ValueError('color 方法需要 --style')
        res = color_transfer_lab(content, style)
    elif method == 'hist':
        if style is None:
            raise ValueError('hist 方法需要 --style')
        res = histogram_match_rgb(content, style)
    elif method == 'texture':
        if style is None:
            raise ValueError('texture 方法需要 --style')
        res = pyramid_texture_blend(content, style, levels)
    elif method == 'blend':
        if style is None:
            raise ValueError('blend 方法需要 --style')
        res = alpha_blend(content, style, alpha)
    elif method == 'neural':
        if style is None:
            raise ValueError('neural 方法需要 --style')
        res = neural_style_transfer(content, style, steps=steps, content_weight=content_weight, style_weight=style_weight)
    else:
        raise ValueError('未知方法')

    save_image(out_path, res)
    print('已保存到', out_path)


def run_pipeline(content_path: str, style_path: str, pipeline: list, out_path: str):
    """pipeline: 方法列表，按顺序作用。支持: sketch, oil, cartoon, color, hist, texture, blend
    对于 blend/texture 用默认参数；如需参数化，可拆分为多次 run。
    """
    img = read_image(content_path)
    style = read_image(style_path) if style_path else None
    for m in pipeline:
        if m == 'sketch':
            img = sketch_effect(img, 'gray', 21, 0, 60, 150)
        elif m == 'oil':
            img = oil_painting_effect(img, 'bilateral', 9, 24, True)
        elif m == 'cartoon':
            img = cartoon_effect(img, 6, 80, 160)
        elif m == 'color':
            if style is None:
                raise ValueError('color 需要 --style')
            img = color_transfer_lab(img, style)
        elif m == 'hist':
            if style is None:
                raise ValueError('hist 需要 --style')
            img = histogram_match_rgb(img, style)
        elif m == 'texture':
            if style is None:
                raise ValueError('texture 需要 --style')
            img = pyramid_texture_blend(img, style, 4)
        elif m == 'blend':
            if style is None:
                raise ValueError('blend 需要 --style')
            img = alpha_blend(img, style, 0.5)
        elif m == 'neural':
            if style is None:
                raise ValueError('neural 需要 --style')
            img = neural_style_transfer(img, style, steps=200)
        else:
            raise ValueError(f'未知方法 {m}')
    save_image(out_path, img)
    print('已保存到', out_path)


def main():
    parser = argparse.ArgumentParser(description='图像特效与风格迁移 CLI')
    sub = parser.add_subparsers(dest='cmd', required=True)

    # 单步骤
    p1 = sub.add_parser('run', help='执行单一方法')
    p1.add_argument('--content', required=True)
    p1.add_argument('--style')
    p1.add_argument('--method', required=True, choices=['sketch', 'oil', 'cartoon', 'color', 'hist', 'texture', 'blend', 'neural'])
    p1.add_argument('--alpha', type=float, default=0.5, help='blend 融合权重')
    p1.add_argument('--levels', type=int, default=4, help='texture 金字塔层数')
    p1.add_argument('--steps', type=int, default=300, help='neural 迭代步数')
    p1.add_argument('--content-weight', type=float, default=1.0)
    p1.add_argument('--style-weight', type=float, default=5.0)
    p1.add_argument('--out', required=True)

    # 多步骤流水线
    p2 = sub.add_parser('pipeline', help='按顺序执行多种方法，逗号分隔')
    p2.add_argument('--content', required=True)
    p2.add_argument('--style')
    p2.add_argument('--steps', required=True, help='如: color,cartoon 或 blend,cartoon')
    p2.add_argument('--out', required=True)

    args = parser.parse_args()

    if args.cmd == 'run':
        run_single(
            args.content,
            args.style,
            args.method,
            args.out,
            args.alpha,
            args.levels,
            args.steps,
            args.content_weight,
            args.style_weight,
        )
    else:
        pipeline = [s.strip() for s in args.steps.split(',') if s.strip()]
        run_pipeline(args.content, args.style, pipeline, args.out)


if __name__ == '__main__':
    main()

