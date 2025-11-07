# 数字图像风格迁移与特效生成系统

## 快速开始

### 环境配置

**重要：编码配置（防止中文乱码）**

本项目使用UTF-8编码，为确保提交信息和中文字符正确显示，请先配置Git编码：

**Windows PowerShell:**
```powershell
# 执行配置脚本
.\setup_git_encoding.ps1

# 或手动配置
git config i18n.commitencoding utf-8
git config i18n.logoutputencoding utf-8
git config core.quotepath false
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

**Git Bash / Linux / Mac:**
```bash
# 执行配置脚本
bash setup_git_encoding.sh

# 或手动配置
git config i18n.commitencoding utf-8
git config i18n.logoutputencoding utf-8
git config core.quotepath false
export LANG=zh_CN.UTF-8
```

详细说明请参考 [ENCODING_FIX_SUMMARY.md](ENCODING_FIX_SUMMARY.md)

### 安装和运行

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

### 深度学习风格迁移（可选）
安装 PyTorch 与 torchvision 后启用（CPU 也可跑，但较慢）：
```bash
pip install torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple
```
使用示例：
```bash
python cli.py run --content 1.png --style 2.png --method neural --steps 300 --style-weight 5.0 --out out_neural.png
```
提示：`--steps` 越大笔触越明显，先从 200~400 尝试；若显得过度，可降低 `--style-weight` 或提高 `--content-weight`。

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
│   ├── encoding.py        # UTF-8编码支持（防止控制台乱码）
│   ├── filters.py         # 素描、油画、卡通
│   ├── color_transfer.py  # Lab/直方图颜色风格迁移
│   ├── texture_transfer.py# 拉普拉斯金字塔融合
│   └── utils.py           # IO/通用工具
├── .gitattributes         # Git文件编码配置
├── setup_git_encoding.ps1 # Git编码配置脚本（Windows）
├── setup_git_encoding.sh  # Git编码配置脚本（Linux/Mac）
├── ENCODING_FIX_SUMMARY.md # 编码问题修复总结
├── requirements.txt
└── README.md
```

## 可选扩展
- 深度学习风格迁移（检测到 PyTorch 则在 GUI 中启用）
- 视频风格化（对帧序列循环处理）
- 自定义风格叠加与混合

## 许可证
仅教学使用。

