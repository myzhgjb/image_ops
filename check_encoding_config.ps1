# 检查Git编码配置脚本
# 用于验证Git编码配置是否正确

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "检查Git编码配置..." -ForegroundColor Cyan
Write-Host ""

$allCorrect = $true

# 检查commitencoding
$commitEncoding = git config --get i18n.commitencoding
if ($commitEncoding -eq "utf-8") {
    Write-Host "✓ commitencoding: $commitEncoding" -ForegroundColor Green
} else {
    Write-Host "✗ commitencoding: $commitEncoding (应该是 utf-8)" -ForegroundColor Red
    $allCorrect = $false
}

# 检查logoutputencoding
$logEncoding = git config --get i18n.logoutputencoding
if ($logEncoding -eq "utf-8") {
    Write-Host "✓ logoutputencoding: $logEncoding" -ForegroundColor Green
} else {
    Write-Host "✗ logoutputencoding: $logEncoding (应该是 utf-8)" -ForegroundColor Red
    $allCorrect = $false
}

# 检查quotepath
$quotepath = git config --get core.quotepath
if ($quotepath -eq "false") {
    Write-Host "✓ quotepath: $quotepath" -ForegroundColor Green
} else {
    Write-Host "✗ quotepath: $quotepath (应该是 false)" -ForegroundColor Red
    $allCorrect = $false
}

# 检查.gitattributes
if (Test-Path .gitattributes) {
    Write-Host "✓ .gitattributes 文件存在" -ForegroundColor Green
} else {
    Write-Host "✗ .gitattributes 文件不存在" -ForegroundColor Red
    $allCorrect = $false
}

Write-Host ""
if ($allCorrect) {
    Write-Host "所有配置正确！" -ForegroundColor Green
} else {
    Write-Host "发现配置问题，请运行 setup_git_encoding.ps1 进行配置" -ForegroundColor Yellow
}

