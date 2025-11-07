# Git提交中文信息指南

## 问题背景

在Windows环境下，如果直接使用PowerShell执行`git commit -m "中文消息"`，可能会因为编码问题导致提交信息在GitHub上显示为乱码。

## 解决方案

### 方法1：使用Git Bash（推荐）

在Git Bash中执行提交命令，Git Bash默认使用UTF-8编码：

```bash
# 在Git Bash中
git commit -m "docs: 添加新功能"
```

### 方法2：使用文件提交（最可靠）

创建一个UTF-8编码的文本文件，然后使用`-F`参数：

```bash
# 1. 创建提交信息文件（确保使用UTF-8编码）
echo "docs: 添加新功能" > commit_msg.txt

# 2. 使用文件提交
git commit -F commit_msg.txt

# 3. 删除临时文件
rm commit_msg.txt
```

**PowerShell版本：**
```powershell
# 创建UTF-8编码的文件
$msg = "docs: 添加新功能"
[System.IO.File]::WriteAllText("commit_msg.txt", $msg, [System.Text.Encoding]::UTF8)

# 提交
git commit -F commit_msg.txt

# 清理
Remove-Item commit_msg.txt
```

### 方法3：配置PowerShell环境（需要每次设置）

在PowerShell中设置编码后提交：

```powershell
# 设置编码
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:LC_ALL = "zh_CN.UTF-8"

# 然后提交（仍可能有问题，不推荐）
git commit -m "docs: 添加新功能"
```

## 验证提交信息

提交后，立即验证提交信息是否正确：

```bash
# 查看最新的提交信息
git log --format="%s" -1 HEAD

# 如果显示乱码，立即修复
git commit --amend -F commit_msg.txt
```

## 推荐的提交流程

1. **编写提交信息到文件**（使用UTF-8编码）
   ```bash
   # Git Bash
   echo "feat(app): 添加新功能" > commit_msg.txt
   ```

2. **使用文件提交**
   ```bash
   git commit -F commit_msg.txt
   ```

3. **验证提交信息**
   ```bash
   git log --format="%s" -1 HEAD
   ```

4. **清理临时文件**
   ```bash
   rm commit_msg.txt
   ```

## 预防措施

### 1. 使用Git钩子

项目已配置`commit-msg`钩子，会自动检查提交信息格式和编码。

### 2. 配置Git编码

确保Git配置了UTF-8编码：

```bash
git config i18n.commitencoding utf-8
git config i18n.logoutputencoding utf-8
git config core.quotepath false
```

### 3. 使用约定式提交格式

使用约定式提交格式可以避免一些问题，格式如下：

```
<type>(<scope>): <subject>

[optional body]

[optional footer]
```

**类型（type）：**
- `feat`: 新功能
- `fix`: 修复bug
- `docs`: 文档更新
- `style`: 代码格式调整
- `refactor`: 代码重构
- `perf`: 性能优化
- `test`: 测试相关
- `chore`: 构建/工具相关

**示例：**
```
feat(app): 添加图像预览功能
fix(cli): 修复参数解析错误
docs: 更新README说明
```

## 如果已经出现乱码

如果提交信息已经出现乱码，可以使用以下方法修复：

### 修复最近一次提交

```bash
# 方法1：使用文件
echo "正确的提交信息" > commit_msg.txt
git commit --amend -F commit_msg.txt

# 方法2：在Git Bash中
git commit --amend -m "正确的提交信息"
```

### 修复历史提交

如果已经推送到远程，需要使用`git filter-branch`重写历史（详见`ENCODING_FIX_SUMMARY.md`）。

## 总结

**最佳实践：**
1. ✅ 在Git Bash中执行提交操作
2. ✅ 使用文件方式提交（`git commit -F`）
3. ✅ 提交后立即验证提交信息
4. ✅ 配置Git使用UTF-8编码
5. ✅ 使用约定式提交格式

**避免：**
1. ❌ 直接在PowerShell中使用`git commit -m "中文"`
2. ❌ 忽略编码配置
3. ❌ 不验证提交信息就直接推送

## 相关文件

- `ENCODING_FIX_SUMMARY.md`: 详细的编码问题修复总结
- `setup_git_encoding.ps1`: Windows Git编码配置脚本
- `setup_git_encoding.sh`: Linux/Mac Git编码配置脚本
- `check_encoding_config.ps1`: 编码配置检查脚本

