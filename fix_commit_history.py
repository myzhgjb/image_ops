#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复Git提交历史中的乱码问题
使用git filter-branch重写提交信息
"""
import subprocess
import sys
import os

# 需要修复的提交哈希和正确的提交信息
COMMIT_FIXES = {
    "5c392b4": "feat(neural_enhanced): 新增增强神经风格迁移方法，多尺度处理、改进损失函数、学习率调度，效果更接近专业工具",
    "77263ff": "feat(app): 添加基于当前预览结果的叠加功能，支持效果链式应用",
    "b5356bb": "feat(app): 集成所有功能到GUI；优化素描效果；添加中文说明",
    "81aa725": "fix(cli): neural 模式参数传递 steps/content_weight/style_weight，修复 NameError",
    "e610422": "feat(cli): 支持 run/pipeline，多图融合与 alpha/纹理融合；移除 bat；补充 README 示例",
}

def run_cmd(cmd, check=True):
    """执行命令"""
    print(f"执行: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='utf-8')
    if check and result.returncode != 0:
        print(f"错误: {result.stderr}")
        sys.exit(1)
    return result.stdout

def main():
    # 确保在正确的目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # 检查当前分支
    branch = run_cmd("git branch --show-current").strip()
    print(f"当前分支: {branch}")
    
    # 获取所有提交
    commits = run_cmd("git log --oneline --all").strip().split('\n')
    
    # 创建修复脚本
    fix_script = """#!/bin/sh
git filter-branch -f --msg-filter '
case "$GIT_COMMIT" in
"""
    
    for commit_hash, correct_msg in COMMIT_FIXES.items():
        # 获取完整的提交哈希
        full_hash = run_cmd(f"git rev-parse {commit_hash}").strip()
        # 转义消息中的特殊字符
        escaped_msg = correct_msg.replace("'", "'\\''")
        fix_script += f"    {full_hash})\n"
        fix_script += f"        echo '{escaped_msg}'\n"
        fix_script += f"        ;;\n"
    
    fix_script += """    *)
        cat
        ;;
esac
' -- --all
"""
    
    # 由于Windows上可能没有bash，我们使用Python直接修改
    print("\n使用git rebase方法修复提交历史...")
    print("警告: 这将重写提交历史，确保已备份！")
    
    # 更简单的方法：使用git commit --amend对最近的提交
    # 或者使用git rebase -i
    print("\n推荐方法：使用交互式rebase手动修复")
    print("执行: git rebase -i bfb8832")
    print("然后将需要修改的提交标记为 'reword' 或 'edit'")
    
    # 或者我们可以创建一个PowerShell脚本
    return

if __name__ == "__main__":
    main()

