#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
重新创建Git提交对象，使用正确的UTF-8编码
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

def run_cmd(cmd, check=True, input_data=None, encoding='utf-8'):
    """执行命令"""
    print(f"执行: {cmd[:100]}...")
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            input=input_data,
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

def get_commit_info(commit_hash):
    """获取提交信息（tree, parent, author, committer）"""
    commit_obj = run_cmd(f"git cat-file -p {commit_hash}")
    if not commit_obj:
        return None
    
    info = {}
    lines = commit_obj.split('\n')
    for line in lines:
        if line.startswith('tree '):
            info['tree'] = line.split()[1]
        elif line.startswith('parent '):
            if 'parent' not in info:
                info['parent'] = []
            info['parent'].append(line.split()[1])
        elif line.startswith('author '):
            info['author'] = line[7:]
        elif line.startswith('committer '):
            info['committer'] = line[10:]
        elif line == '':
            break
    
    return info

def rebuild_commit(old_hash, new_msg):
    """重新创建提交对象"""
    info = get_commit_info(old_hash)
    if not info:
        print(f"无法获取提交 {old_hash} 的信息")
        return None
    
    # 构建新的提交对象内容
    commit_content = f"tree {info['tree']}\n"
    if 'parent' in info:
        for parent in info['parent']:
            commit_content += f"parent {parent}\n"
    commit_content += f"author {info['author']}\n"
    commit_content += f"committer {info['committer']}\n"
    commit_content += f"\n{new_msg}\n"
    
    # 使用git hash-object创建新对象
    new_hash = run_cmd("git hash-object -t commit -w --stdin", input_data=commit_content, encoding='utf-8')
    return new_hash

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("重新创建Git提交对象...")
    print("这将创建新的提交对象，使用正确的UTF-8编码")
    
    # 从最旧的提交开始修复
    commit_order = ["e610422", "81aa725", "b5356bb", "77263ff", "5c392b4"]
    
    # 映射：旧哈希 -> 新哈希
    hash_map = {}
    
    for commit_hash in commit_order:
        new_msg = COMMIT_FIXES[commit_hash]
        print(f"\n处理提交 {commit_hash}...")
        
        info = get_commit_info(commit_hash)
        if not info:
            continue
        
        # 如果有父提交，检查是否需要更新父提交哈希
        if 'parent' in info:
            new_parents = []
            for parent in info['parent']:
                # 检查父提交是否也需要修复
                short_parent = parent[:7]
                if short_parent in hash_map:
                    new_parents.append(hash_map[short_parent])
                else:
                    # 检查父提交是否在修复列表中
                    found = False
                    for fix_hash in commit_order:
                        if parent.startswith(fix_hash):
                            # 这个父提交也需要修复，但我们按顺序处理
                            new_parents.append(parent)
                            found = True
                            break
                    if not found:
                        new_parents.append(parent)
            info['parent'] = new_parents
        
        # 重新创建提交
        commit_content = f"tree {info['tree']}\n"
        if 'parent' in info:
            for parent in info['parent']:
                commit_content += f"parent {parent}\n"
        commit_content += f"author {info['author']}\n"
        commit_content += f"committer {info['committer']}\n"
        commit_content += f"\n{new_msg}\n"
        
        new_hash = run_cmd("git hash-object -t commit -w --stdin", input_data=commit_content, encoding='utf-8')
        if new_hash:
            hash_map[commit_hash] = new_hash
            print(f"  -> 新提交: {new_hash}")
        else:
            print(f"  -> 失败")
    
    if hash_map:
        print("\n已创建新的提交对象")
        print("现在需要使用git filter-branch或git replace来替换旧的提交")
        print("\n使用git replace方法（更安全）:")
        for old_hash, new_hash in hash_map.items():
            print(f"  git replace {old_hash} {new_hash}")
    else:
        print("未能创建新的提交对象")

if __name__ == "__main__":
    main()

