#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
重写Git提交历史，修复乱码的提交信息
使用方法: python rewrite_commits.py
"""
import subprocess
import sys
import os
import tempfile

# 需要修复的提交哈希和正确的提交信息（按时间顺序，从旧到新）
COMMIT_FIXES = [
    ("e610422", "feat(cli): 支持 run/pipeline，多图融合与 alpha/纹理融合；移除 bat；补充 README 示例"),
    ("81aa725", "fix(cli): neural 模式参数传递 steps/content_weight/style_weight，修复 NameError"),
    ("b5356bb", "feat(app): 集成所有功能到GUI；优化素描效果；添加中文说明"),
    ("77263ff", "feat(app): 添加基于当前预览结果的叠加功能，支持效果链式应用"),
    ("5c392b4", "feat(neural_enhanced): 新增增强神经风格迁移方法，多尺度处理、改进损失函数、学习率调度，效果更接近专业工具"),
]

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
            print(f"错误输出: {result.stderr}")
            return None
        return result.stdout.strip()
    except Exception as e:
        print(f"执行命令时出错: {e}")
        return None

def get_full_hash(short_hash):
    """获取完整的提交哈希"""
    return run_cmd(f"git rev-parse {short_hash}")

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("=" * 60)
    print("Git提交历史乱码修复工具")
    print("=" * 60)
    
    # 检查是否有未提交的更改
    status = run_cmd("git status --porcelain")
    if status:
        print("警告: 检测到未提交的更改!")
        print("请先提交或暂存所有更改后再运行此脚本")
        response = input("是否继续? (y/n): ")
        if response.lower() != 'y':
            return
    
    # 获取当前分支
    branch = run_cmd("git branch --show-current")
    print(f"当前分支: {branch}")
    
    # 检查是否在main分支
    if branch != "main":
        print(f"警告: 当前不在main分支，而是在 {branch} 分支")
        response = input("是否继续? (y/n): ")
        if response.lower() != 'y':
            return
    
    print("\n将使用git filter-branch重写提交历史")
    print("这将修改所有分支的历史，请确保已备份!")
    response = input("确认继续? (yes/no): ")
    if response.lower() != 'yes':
        print("已取消")
        return
    
    # 创建filter-branch脚本
    filter_script = """#!/bin/sh
case "$GIT_COMMIT" in
"""
    
    for short_hash, correct_msg in COMMIT_FIXES:
        full_hash = get_full_hash(short_hash)
        if not full_hash:
            print(f"警告: 无法找到提交 {short_hash}，跳过")
            continue
        
        # 转义消息中的单引号
        escaped_msg = correct_msg.replace("'", "'\\''")
        filter_script += f"    {full_hash})\n"
        filter_script += f"        echo '{escaped_msg}'\n"
        filter_script += f"        ;;\n"
    
    filter_script += """    *)
        cat
        ;;
esac
"""
    
    # 保存脚本
    script_file = os.path.join(os.getcwd(), "filter-msg.sh")
    try:
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(filter_script)
        
        # 设置执行权限（Windows上可能不需要）
        if sys.platform != 'win32':
            os.chmod(script_file, 0o755)
        
        print(f"\n已创建filter脚本: {script_file}")
        print("\n执行git filter-branch...")
        
        # 执行filter-branch
        cmd = f'git filter-branch -f --msg-filter "bash {script_file}" -- --all'
        result = run_cmd(cmd, check=False)
        
        if result is not None:
            print("提交历史已重写!")
            print("\n下一步:")
            print("1. 检查提交历史: git log --oneline")
            print("2. 如果满意，强制推送到远程: git push --force-with-lease origin main")
            print("3. 警告: 强制推送会覆盖远程历史，请确保团队成员已同步!")
        else:
            print("执行失败，可能需要手动执行:")
            print(f"  git filter-branch -f --msg-filter 'bash {script_file}' -- --all")
    
    except Exception as e:
        print(f"错误: {e}")
        print("\n备用方案: 使用交互式rebase")
        print("执行: git rebase -i bfb8832")
        print("然后将需要修改的提交标记为 'reword'")

if __name__ == "__main__":
    main()

