# 数字图像风格迁移与特效生成系统

## 快速开始

1) 安装依赖（已在虚拟环境中）：

```bash
pip install -r requirements.txt
```

2) 运行 GUI：

```bash
python app.py
```

3) CLI 示例（任意两图融合/转换，均为 Python 命令，无需 bat）：

- 单步骤：色彩风格迁移（`2.png` → `1.png`）
```bash
python cli.py run --content 1.png --style 2.png --method color --out out_color_from_2.png
```

- 单步骤：金字塔纹理融合
```bash
python cli.py run --content 1.png --style 2.png --method texture --levels 4 --out out_texture_blend.png
```

- 单步骤：alpha 线性融合（任意两图）
```bash
python cli.py run --content 1.png --style 2.png --method blend --alpha 0.6 --out out_alpha_0_6.png
```

- 多步骤流水线：先做色彩风格迁移，再卡通化
```bash
python cli.py pipeline --content 1.png --style 2.png --steps color,cartoon --out out_color_then_cartoon.png
```

## 功能模块
- 基于滤波器的艺术特效：素描、油画、卡通
- 基于传统方法的风格迁移：Lab 颜色分布匹配、直方图匹配
- 纹理合成：拉普拉斯金字塔融合
- 交互式 GUI：参数滑块、实时预览、批量处理与保存
- CLI：批量命令行处理，快速复现实验

## 目录结构
```
.
├── app.py                 # Tkinter GUI 主程序
├── cli.py                 # 命令行入口（支持 run/pipeline、多图融合）
├── batch.py               # 批处理逻辑
├── image_ops/
│   ├── __init__.py
│   ├── filters.py         # 素描、油画、卡通
│   ├── color_transfer.py  # Lab/直方图颜色风格迁移
│   ├── texture_transfer.py# 拉普拉斯金字塔融合
│   └── utils.py           # IO/通用工具
├── requirements.txt
└── README.md
```

## 可选扩展
- 深度学习风格迁移（检测到 PyTorch 则在 GUI 中启用）
- 视频风格化（对帧序列循环处理）
- 自定义风格叠加与混合

## 许可证
仅教学使用。

