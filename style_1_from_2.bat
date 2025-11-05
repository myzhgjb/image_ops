@echo off
REM 将 2.png 的色彩风格迁移到 1.png
python cli.py --content 1.png --style 2.png --method color --out out_1_from_2.png
echo 已生成 out_1_from_2.png

