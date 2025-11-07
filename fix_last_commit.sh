#!/bin/bash
# 修复最后一次提交的信息

export LANG=zh_CN.UTF-8
export LC_ALL=zh_CN.UTF-8

CURRENT_HASH=$(git rev-parse HEAD)

git filter-branch -f --msg-filter "
if [ \"\$GIT_COMMIT\" = \"$CURRENT_HASH" ]; then
    echo 'docs: 添加Git提交中文信息指南'
else
    cat
fi
" HEAD~1..HEAD

echo "修复完成，请检查: git log --format=\"%s\" -1 HEAD"

