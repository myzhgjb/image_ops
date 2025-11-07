# 直接修复Git提交历史中的乱码问题
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

Write-Host "开始修复Git提交历史中的乱码问题..." -ForegroundColor Green

# 需要修复的提交哈希和正确的提交信息
$fixes = @{
    "e610422" = "feat(cli): 支持 run/pipeline，多图融合与 alpha/纹理融合；移除 bat；补充 README 示例"
    "81aa725" = "fix(cli): neural 模式参数传递 steps/content_weight/style_weight，修复 NameError"
    "b5356bb" = "feat(app): 集成所有功能到GUI；优化素描效果；添加中文说明"
    "77263ff" = "feat(app): 添加基于当前预览结果的叠加功能，支持效果链式应用"
    "5c392b4" = "feat(neural_enhanced): 新增增强神经风格迁移方法，多尺度处理、改进损失函数、学习率调度，效果更接近专业工具"
}

# 创建filter-branch脚本内容
$filterScript = @'
#!/bin/sh
case "$GIT_COMMIT" in
'@

foreach ($hash in $fixes.Keys) {
    $fullHash = git rev-parse $hash 2>$null
    if ($fullHash) {
        $msg = $fixes[$hash]
        # 转义单引号
        $escapedMsg = $msg -replace "'", "'\''"
        $filterScript += @"

    $fullHash)
        echo '$escapedMsg'
        ;;
"@
    }
}

$filterScript += @'
    *)
        cat
        ;;
esac
'@

# 保存到文件
$scriptFile = "filter-msg-commit.sh"
$filterScript | Out-File -FilePath $scriptFile -Encoding UTF8 -NoNewline

Write-Host "已创建filter脚本: $scriptFile" -ForegroundColor Cyan

# 查找bash
$bashPath = $null
$possiblePaths = @(
    "C:\Program Files\Git\usr\bin\bash.exe",
    "C:\Program Files (x86)\Git\usr\bin\bash.exe",
    "$env:ProgramFiles\Git\usr\bin\bash.exe",
    "$env:ProgramFiles(x86)\Git\usr\bin\bash.exe"
)

foreach ($path in $possiblePaths) {
    if (Test-Path $path) {
        $bashPath = $path
        break
    }
}

if (-not $bashPath) {
    $bashPath = "bash"
}

Write-Host "使用bash: $bashPath" -ForegroundColor Cyan

# 执行git filter-branch
Write-Host "`n执行git filter-branch..." -ForegroundColor Yellow
Write-Host "警告: 这将重写所有分支的提交历史!" -ForegroundColor Red

$filterCmd = "git filter-branch -f --msg-filter `"$bashPath $scriptFile`" -- --all"
Write-Host "执行命令: $filterCmd" -ForegroundColor Cyan

try {
    Invoke-Expression $filterCmd
    Write-Host "`n提交历史已重写!" -ForegroundColor Green
    Write-Host "`n下一步操作:" -ForegroundColor Yellow
    Write-Host "1. 检查提交历史: git log --oneline -10" -ForegroundColor White
    Write-Host "2. 如果满意，强制推送到远程: git push --force-with-lease origin main" -ForegroundColor White
    Write-Host "3. 警告: 强制推送会覆盖远程历史!" -ForegroundColor Red
} catch {
    Write-Host "执行失败: $_" -ForegroundColor Red
    Write-Host "`n可能需要手动执行以下命令:" -ForegroundColor Yellow
    Write-Host "  git filter-branch -f --msg-filter `"bash $scriptFile`" -- --all" -ForegroundColor White
}

