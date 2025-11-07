#!/bin/sh
case "$GIT_COMMIT" in
    e6104222fdae1d410685357f0d9f626ae8809ead)
        echo 'feat(cli): 支持 run/pipeline，多图融合与 alpha/纹理融合；移除 bat；补充 README 示例'
        ;;
    81aa725f743b9e6ea1004a964b9512be15873501)
        echo 'fix(cli): neural 模式参数传递 steps/content_weight/style_weight，修复 NameError'
        ;;
    b5356bbf017aa6f3904297b8f0ba60203ef24520)
        echo 'feat(app): 集成所有功能到GUI；优化素描效果；添加中文说明'
        ;;
    77263ffa6f7052d3444b5c7eff8c2910550fb3cb)
        echo 'feat(app): 添加基于当前预览结果的叠加功能，支持效果链式应用'
        ;;
    5c392b40f7045a02aef3a647dd445abddf2a6721)
        echo 'feat(neural_enhanced): 新增增强神经风格迁移方法，多尺度处理、改进损失函数、学习率调度，效果更接近专业工具'
        ;;
    *)
        cat
        ;;
esac
