#!/bin/sh

# 检查环境变量是否存在，如果存在，就把它写入到文件中
if [ -n "$GEMINI_CREDENTIALS_JSON" ]; then
  echo "$GEMINI_CREDENTIALS_JSON" > "$GOOGLE_APPLICATION_CREDENTIALS"
fi

# 执行 Dockerfile 中原本的 CMD 命令
exec "$@"
