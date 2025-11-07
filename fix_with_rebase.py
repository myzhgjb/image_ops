#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用git rebase自动修复提交历史中的乱码问题
"""
import subprocess
import sys
import os
import tempfile

# 需要修复的提交哈希和正确的提交信息
COMMIT_FIXES = {
    "e610422": "feat(cli): 支持 run/pipeline，多图融合与 alpha/纹理融合；移除 bat；补充 README 示例",
    "81aa725": "fix(cli): neural 模式参数传递 steps/content_weight/style_weight，修复 NameError",
    "b5356bb": "feat(app): 集成所有功能到GUI；优化素描效果；添加中文说明",
    "77263ff": "feat(app): 添加基于当前预览结果的叠加功能，支持效果链式应用",
    "5c392b4": "feat(neural_enhanced): 新增增强神经风格迁移方法，多尺度处理、改进损失函数、学习率调度，效果更接近专业工具",
}

def run_cmd(cmd, check=True, encoding='utf-8'):
    """执行命令"""
    print(f"执行: {cmd}")
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            encoding=encoding,
            errors='replace'
        )
        if check and result.returncode != 0:
            print(f"错误: {result.stderr}")
            return None
        return result.stdout.strip()
    except Exception as e:
        print(f"错误: {e}")
        return None

def create_editor_script(commits_to_fix):
    """创建自动化rebase editor脚本"""
    script = """#!/bin/sh
# 自动rebase脚本
"""
    for commit_hash in commits_to_fix:
        script += f'git commit --amend -m "{COMMIT_FIXES[commit_hash]}"\n'
    return script

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("使用git filter-branch修复提交历史...")
    print("这次使用更直接的方法")
    
    # 创建filter脚本，使用单引号避免转义问题
    filter_script_lines = ['#!/bin/sh', 'case "$GIT_COMMIT" in']
    
    for short_hash, msg in COMMIT_FIXES.items():
        full_hash = run_cmd(f"git rev-parse {short_hash}")
        if full_hash:
            # 使用printf而不是echo，更好地处理特殊字符
            filter_script_lines.append(f"    {full_hash})")
            # 转义消息中的特殊字符
            escaped = msg.replace("\\", "\\\\").replace("$", "\\$").replace("`", "\\`")
            filter_script_lines.append(f'        printf "%s\\n" "{escaped}"')
            filter_script_lines.append("        ;;")
    
    filter_script_lines.append("    *)")
    filter_script_lines.append("        cat")
    filter_script_lines.append("        ;;")
    filter_script_lines.append("esac")
    
    script_content = "\n".join(filter_script_lines)
    
    script_file = "filter-msg-fix.sh"
    with open(script_file, 'w', encoding='utf-8', newline='\n') as f:
        f.write(script_content)
    
    print(f"已创建脚本: {script_file}")
    print("执行git filter-branch...")
    
    # 查找bash
    git_path = run_cmd("where git")
    if git_path:
        git_dir = os.path.dirname(git_path)
        bash_paths = [
            os.path.join(git_dir, "..", "usr", "bin", "bash.exe"),
            os.path.join(os.path.dirname(git_dir), "usr", "bin", "bash.exe"),
            r"C:\software\git\Git\usr\bin\bash.exe",
        ]
        bash_path = None
        for path in bash_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                bash_path = abs_path
                break
        
        if bash_path:
            script_abs = os.path.abspath(script_file).replace("\\", "/")
            cmd = f'git filter-branch -f --msg-filter "{bash_path} {script_abs}" -- --all'
            print(f"执行: {cmd}")
            result = run_cmd(cmd, check=False)
            
            # 检查结果
            test_hash = "5c392b4"
            test_msg = run_cmd(f'git log --format="%s" -1 {test_hash}')
            if test_msg and "新增增强" in test_msg:
                print("✓ 提交历史修复成功!")
            else:
                print("✗ 修复可能未成功，提交信息仍为:", test_msg)
                print("请手动检查")
        else:
            print("未找到bash，请手动执行")
    else:
        print("未找到git")

if __name__ == "__main__":
    main()

