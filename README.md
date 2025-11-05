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

3) 命令行示例：将 `2.png` 的色彩风格迁移到 `1.png`：

```bash
python cli.py --content 1.png --style 2.png --method color --out out_color_transfer.png
```

同样可用 `--method sketch|oil|cartoon|texture` 等。

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
├── cli.py                 # 命令行入口（含 2.png -> 1.png 示例）
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

