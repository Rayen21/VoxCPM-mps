#!/bin/bash

# ==============================
# 配置区
# ==============================
CONDA_BASE="$HOME/miniforge3"
CONDA_ENV="voxcpm" #根据自己实际的环境名修改
PROJECT_DIR="$HOME/VoxCPM" #根据自己实际项目放置目录修改
PORT=8000
URL="http://127.0.0.1:${PORT}"

echo "=============================="
echo "🚀 VoxCPM 启动中"
echo "=============================="

# ==============================
# 初始化 conda
# ==============================
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    echo "❌ 找不到 conda.sh"
    read -n 1
    exit 1
fi

conda activate "$CONDA_ENV" || exit 1
cd "$PROJECT_DIR" || exit 1

# ==============================
# 🔥 启动前清理端口
# ==============================
echo "🧹 清理端口 $PORT ..."
lsof -ti :$PORT | xargs kill -9 2>/dev/null \
    && echo "✓ 端口 $PORT 已释放" \
    || echo "⚠️ 端口 $PORT 未被占用"

# ==============================
# 延迟打开浏览器（后台）
# ==============================
(
  echo "⏳ 等待服务启动..."
  while ! lsof -i tcp:$PORT >/dev/null 2>&1; do
      sleep 0.5
  done
  echo "🌐 打开浏览器：$URL"
  open "$URL"
) &

# ==============================
# 前台启动（关键）
# ==============================
echo "📜 服务运行中，按 Ctrl+C 退出"
python main.py