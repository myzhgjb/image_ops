#!/bin/bash
# 修复提交信息的脚本

git filter-branch -f --msg-filter '
if [ "$GIT_COMMIT" = "5c392b4" ]; then
    echo "feat(neural_enhanced): 新增增强神经风格迁移方法，多尺度处理、改进损失函数、学习率调度，效果更接近专业工具"
elif [ "$GIT_COMMIT" = "77263ff" ]; then
    echo "feat(app): 添加基于当前预览结果的叠加功能，支持效果链式应用"
elif [ "$GIT_COMMIT" = "b5356bb" ]; then
    echo "feat(app): 集成所有功能到GUI；优化素描效果；添加中文说明"
elif [ "$GIT_COMMIT" = "81aa725" ]; then
    echo "fix(cli): neural 模式参数传递 steps/content_weight/style_weight，修复 NameError"
elif [ "$GIT_COMMIT" = "e610422" ]; then
    echo "feat(cli): 支持 run/pipeline，多图融合与 alpha/纹理融合；移除 bat；补充 README 示例"
else
    cat
fi
' HEAD~6..HEAD

