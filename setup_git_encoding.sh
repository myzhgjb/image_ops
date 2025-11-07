#!/bin/bash
# Git编码配置脚本
# 用于设置Git使用UTF-8编码，防止中文乱码问题

set -e

echo "=========================================="
echo "Git UTF-8编码配置脚本"
echo "=========================================="
echo ""

# 检查Git是否安装
if ! command -v git >/dev/null 2>&1; then
    echo "错误: 未找到Git，请先安装Git"
    exit 1
fi

echo "1. 配置Git使用UTF-8编码..."

# 配置本地仓库
git config i18n.commitencoding utf-8
git config i18n.logoutputencoding utf-8
git config core.quotepath false

echo "   ✓ 本地仓库配置完成"

# 询问是否配置全局
read -p "2. 是否同时配置全局Git编码设置？(y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    git config --global i18n.commitencoding utf-8
    git config --global i18n.logoutputencoding utf-8
    git config --global core.quotepath false
    echo "   ✓ 全局配置完成"
else
    echo "   ⏭ 跳过全局配置"
fi

echo ""
echo "3. 设置Git钩子权限..."

# 设置钩子文件为可执行
if [ -f ".git/hooks/commit-msg" ]; then
    chmod +x .git/hooks/commit-msg
    echo "   ✓ commit-msg钩子已设置"
fi

if [ -f ".git/hooks/pre-commit" ]; then
    chmod +x .git/hooks/pre-commit
    echo "   ✓ pre-commit钩子已设置"
fi

echo ""
echo "4. 验证配置..."

# 显示当前配置
echo "   当前Git编码配置:"
git config --get i18n.commitencoding && echo "   ✓ commitencoding: $(git config --get i18n.commitencoding)" || echo "   ✗ commitencoding: 未设置"
git config --get i18n.logoutputencoding && echo "   ✓ logoutputencoding: $(git config --get i18n.logoutputencoding)" || echo "   ✗ logoutputencoding: 未设置"
git config --get core.quotepath && echo "   ✓ quotepath: $(git config --get core.quotepath)" || echo "   ✗ quotepath: 未设置"

echo ""
echo "=========================================="
echo "配置完成！"
echo "=========================================="
echo ""
echo "提示:"
echo "  - 提交信息请使用UTF-8编码"
echo "  - 在Windows上，建议使用Git Bash或配置PowerShell使用UTF-8"
echo "  - 如果使用PowerShell，可以设置: [Console]::OutputEncoding = [System.Text.Encoding]::UTF8"
echo ""

