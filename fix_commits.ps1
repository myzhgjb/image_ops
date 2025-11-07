# 修复乱码提交信息的脚本
$commits = @(
    @{hash="5c392b4"; msg="feat(neural_enhanced): 新增增强神经风格迁移方法，多尺度处理、改进损失函数、学习率调度，效果更接近专业工具"},
    @{hash="77263ff"; msg="feat(app): 添加基于当前预览结果的叠加功能，支持效果链式应用"},
    @{hash="b5356bb"; msg="feat(app): 集成所有功能到GUI；优化素描效果；添加中文说明"},
    @{hash="81aa725"; msg="fix(cli): neural 模式参数传递 steps/content_weight/style_weight，修复 NameError"},
    @{hash="e610422"; msg="feat(cli): 支持 run/pipeline，多图融合与 alpha/纹理融合；移除 bat；补充 README 示例"}
)

foreach ($commit in $commits) {
    Write-Host "修复提交 $($commit.hash)..."
    git rebase -i $commit.hash^
}

