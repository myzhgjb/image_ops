# Git编码配置脚本 (PowerShell版本)
# 用于设置Git使用UTF-8编码，防止中文乱码问题

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Git UTF-8编码配置脚本" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# 检查Git是否安装
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "错误: 未找到Git，请先安装Git" -ForegroundColor Red
    exit 1
}

Write-Host "1. 配置Git使用UTF-8编码..." -ForegroundColor Yellow

# 配置本地仓库
git config i18n.commitencoding utf-8
git config i18n.logoutputencoding utf-8
git config core.quotepath false

Write-Host "   ✓ 本地仓库配置完成" -ForegroundColor Green

# 询问是否配置全局
$response = Read-Host "2. 是否同时配置全局Git编码设置？(y/n)"
if ($response -eq "y" -or $response -eq "Y") {
    git config --global i18n.commitencoding utf-8
    git config --global i18n.logoutputencoding utf-8
    git config --global core.quotepath false
    Write-Host "   ✓ 全局配置完成" -ForegroundColor Green
} else {
    Write-Host "   ⏭ 跳过全局配置" -ForegroundColor Gray
}

Write-Host ""
Write-Host "3. 配置PowerShell使用UTF-8编码..." -ForegroundColor Yellow

# 设置PowerShell输出编码
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
Write-Host "   ✓ PowerShell编码已设置" -ForegroundColor Green

Write-Host ""
Write-Host "4. 设置Git钩子..." -ForegroundColor Yellow

# 设置钩子文件（如果存在）
$hooks = @(".git/hooks/commit-msg", ".git/hooks/pre-commit")
foreach ($hook in $hooks) {
    if (Test-Path $hook) {
        # 在Windows上，Git钩子应该使用LF行尾
        Write-Host "   ✓ $hook 已存在" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "5. 验证配置..." -ForegroundColor Yellow

# 显示当前配置
Write-Host "   当前Git编码配置:" -ForegroundColor Cyan
$commitEncoding = git config --get i18n.commitencoding
$logEncoding = git config --get i18n.logoutputencoding
$quotepath = git config --get core.quotepath

if ($commitEncoding) {
    Write-Host "   ✓ commitencoding: $commitEncoding" -ForegroundColor Green
} else {
    Write-Host "   ✗ commitencoding: 未设置" -ForegroundColor Red
}

if ($logEncoding) {
    Write-Host "   ✓ logoutputencoding: $logEncoding" -ForegroundColor Green
} else {
    Write-Host "   ✗ logoutputencoding: 未设置" -ForegroundColor Red
}

if ($quotepath) {
    Write-Host "   ✓ quotepath: $quotepath" -ForegroundColor Green
} else {
    Write-Host "   ✗ quotepath: 未设置" -ForegroundColor Red
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "配置完成！" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "提示:" -ForegroundColor Yellow
Write-Host "  - 提交信息请使用UTF-8编码"
Write-Host "  - 建议在Git Bash中执行Git操作，或者在PowerShell中设置UTF-8编码"
Write-Host "  - 可以在PowerShell配置文件中添加以下内容以永久设置编码:"
Write-Host "    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8" -ForegroundColor Gray
Write-Host "    `$env:PYTHONIOENCODING = 'utf-8'" -ForegroundColor Gray
Write-Host ""

