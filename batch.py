import os
from typing import Callable, Optional

from image_ops import read_image, save_image


def process_directory(in_dir: str, out_dir: str, func: Callable[[any], any]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for name in os.listdir(in_dir):
        if name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            ip = os.path.join(in_dir, name)
            op = os.path.join(out_dir, name)
            try:
                img = read_image(ip)
                res = func(img)
                save_image(op, res)
            except Exception as e:
                print('处理失败:', ip, e)

