import argparse
from image_ops import (
    read_image, save_image,
    sketch_effect, oil_painting_effect, cartoon_effect,
    color_transfer_lab, histogram_match_rgb,
    pyramid_texture_blend
)


def main():
    parser = argparse.ArgumentParser(description='图像特效与风格迁移 CLI')
    parser.add_argument('--content', required=True, help='内容图路径')
    parser.add_argument('--style', help='风格/纹理图路径（部分方法需要）')
    parser.add_argument('--method', required=True, choices=['sketch', 'oil', 'cartoon', 'color', 'hist', 'texture'])
    parser.add_argument('--out', required=True, help='输出路径')
    args = parser.parse_args()

    content = read_image(args.content)
    style = read_image(args.style) if args.style else None

    if args.method == 'sketch':
        res = sketch_effect(content, 'gray', 21, 0, 60, 150)
    elif args.method == 'oil':
        res = oil_painting_effect(content, 'bilateral', 9, 24, True)
    elif args.method == 'cartoon':
        res = cartoon_effect(content, 6, 80, 160)
    elif args.method == 'color':
        if style is None:
            raise ValueError('color 方法需要 --style')
        res = color_transfer_lab(content, style)
    elif args.method == 'hist':
        if style is None:
            raise ValueError('hist 方法需要 --style')
        res = histogram_match_rgb(content, style)
    elif args.method == 'texture':
        if style is None:
            raise ValueError('texture 方法需要 --style')
        res = pyramid_texture_blend(content, style, 4)
    else:
        raise ValueError('未知方法')

    save_image(args.out, res)
    print('已保存到', args.out)


if __name__ == '__main__':
    main()

