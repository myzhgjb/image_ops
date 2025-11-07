# PowerShell脚本：修复Git提交历史中的乱码问题
# 使用git filter-branch重写提交信息

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# 需要修复的提交哈希和正确的提交信息
$commitFixes = @{
    "5c392b4" = "feat(neural_enhanced): 新增增强神经风格迁移方法，多尺度处理、改进损失函数、学习率调度，效果更接近专业工具"
    "77263ff" = "feat(app): 添加基于当前预览结果的叠加功能，支持效果链式应用"
    "b5356bb" = "feat(app): 集成所有功能到GUI；优化素描效果；添加中文说明"
    "81aa725" = "fix(cli): neural 模式参数传递 steps/content_weight/style_weight，修复 NameError"
    "e610422" = "feat(cli): 支持 run/pipeline，多图融合与 alpha/纹理融合；移除 bat；补充 README 示例"
}

Write-Host "开始修复Git提交历史中的乱码问题..." -ForegroundColor Green
Write-Host "警告: 这将重写提交历史！" -ForegroundColor Yellow

# 获取当前分支
$currentBranch = git branch --show-current
Write-Host "当前分支: $currentBranch" -ForegroundColor Cyan

# 创建一个临时脚本文件用于git filter-branch
$filterScript = @"
#!/bin/sh
case `$GIT_COMMIT in
"@

foreach ($hash in $commitFixes.Keys) {
    # 获取完整的提交哈希
    $fullHash = git rev-parse $hash 2>$null
    if ($fullHash) {
        $msg = $commitFixes[$hash]
        # 转义单引号和特殊字符
        $escapedMsg = $msg -replace "'", "'\''" -replace "`n", " "
        $filterScript += @"
    $fullHash)
        echo '$escapedMsg'
        ;;
"@
    }
}

$filterScript += @"
    *)
        cat
        ;;
esac
"@

# 保存脚本到临时文件
$scriptFile = Join-Path $PSScriptRoot "filter-msg.sh"
$filterScript | Out-File -FilePath $scriptFile -Encoding UTF8 -NoNewline

Write-Host "`n使用git filter-branch重写提交历史..." -ForegroundColor Cyan
Write-Host "这将修改所有分支的历史，请确保已备份！" -ForegroundColor Yellow

# 执行git filter-branch
# 注意：在Windows上，需要确保有bash或git-bash
$bashPath = "bash"
if (Get-Command git -ErrorAction SilentlyContinue) {
    $gitPath = (Get-Command git).Source
    $gitDir = Split-Path $gitPath
    $bashPath = Join-Path $gitDir "usr\bin\bash.exe"
    if (-not (Test-Path $bashPath)) {
        $bashPath = "bash"
    }
}

# 使用更简单的方法：逐个修复提交
Write-Host "`n使用交互式方法修复提交..." -ForegroundColor Cyan
Write-Host "请按照以下步骤操作：" -ForegroundColor Yellow
Write-Host "1. 执行: git rebase -i bfb8832" -ForegroundColor White
Write-Host "2. 将需要修改的提交前的 'pick' 改为 'reword' 或 'edit'" -ForegroundColor White
Write-Host "3. 保存并退出，然后输入正确的提交信息" -ForegroundColor White

# 或者，我们可以使用git commit --amend逐个修复
Write-Host "`n或者使用以下命令逐个修复：" -ForegroundColor Cyan
foreach ($hash in $commitFixes.Keys) {
    $msg = $commitFixes[$hash]
    Write-Host "git rebase -i ${hash}^" -ForegroundColor White
    Write-Host "  -> 将提交标记为 'reword'，然后输入: $msg" -ForegroundColor Gray
}

Write-Host "`n脚本执行完成。请手动执行上述步骤来修复提交历史。" -ForegroundColor Green

