#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动修复Git提交历史中的乱码问题
使用git rebase来自动修复提交信息
"""
import subprocess
import sys
import os
import tempfile

# 需要修复的提交哈希和正确的提交信息（从旧到新）
COMMIT_FIXES = [
    ("e610422", "feat(cli): 支持 run/pipeline，多图融合与 alpha/纹理融合；移除 bat；补充 README 示例"),
    ("81aa725", "fix(cli): neural 模式参数传递 steps/content_weight/style_weight，修复 NameError"),
    ("b5356bb", "feat(app): 集成所有功能到GUI；优化素描效果；添加中文说明"),
    ("77263ff", "feat(app): 添加基于当前预览结果的叠加功能，支持效果链式应用"),
    ("5c392b4", "feat(neural_enhanced): 新增增强神经风格迁移方法，多尺度处理、改进损失函数、学习率调度，效果更接近专业工具"),
]

def run_cmd(cmd, check=True):
    """执行命令"""
    print(f"执行: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if check and result.returncode != 0:
        print(f"错误: {result.stderr}")
        return None
    return result.stdout.strip()

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("=" * 60)
    print("自动修复Git提交历史中的乱码问题")
    print("=" * 60)
    
    # 使用git filter-branch的方法
    print("\n方法1: 使用git filter-branch (推荐)")
    print("这将重写所有分支的提交历史")
    
    # 创建filter脚本
    filter_content = '#!/bin/sh\ncase "$GIT_COMMIT" in\n'
    
    for short_hash, msg in COMMIT_FIXES:
        full_hash = run_cmd(f"git rev-parse {short_hash}")
        if full_hash:
            escaped_msg = msg.replace("'", "'\\''")
            filter_content += f"    {full_hash})\n"
            filter_content += f"        echo '{escaped_msg}'\n"
            filter_content += f"        ;;\n"
    
    filter_content += """    *)
        cat
        ;;
esac
"""
    
    script_file = "filter-msg.sh"
    with open(script_file, 'w', encoding='utf-8') as f:
        f.write(filter_content)
    
    print(f"\n已创建filter脚本: {script_file}")
    print("\n执行git filter-branch...")
    
    # 尝试使用Git自带的bash
    git_path = run_cmd("where git")
    if git_path:
        git_dir = os.path.dirname(git_path)
        bash_paths = [
            os.path.join(git_dir, "..", "usr", "bin", "bash.exe"),
            os.path.join(os.path.dirname(git_dir), "usr", "bin", "bash.exe"),
        ]
        bash_path = None
        for path in bash_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                bash_path = abs_path
                break
        
        if bash_path:
            print(f"找到bash: {bash_path}")
            script_abs = os.path.abspath(script_file)
            cmd = f'git filter-branch -f --msg-filter "{bash_path} {script_abs}" -- --all'
            result = run_cmd(cmd, check=False)
            if result is not None:
                print("提交历史已重写!")
                print("\n请检查: git log --oneline -10")
                print("如果满意，执行: git push --force-with-lease origin main")
                return
    
    # 如果bash不可用，提供手动指导
    print("\n无法自动执行，请手动执行以下步骤:")
    print(f"1. 打开Git Bash")
    print(f"2. 执行: git filter-branch -f --msg-filter 'bash {script_file}' -- --all")
    print("3. 检查结果: git log --oneline -10")
    print("4. 如果满意，执行: git push --force-with-lease origin main")

if __name__ == "__main__":
    main()

