#!/bin/bash
# VoCoType IBus 引擎启动脚本
# 在启动 Python 前设置正确的 CUDA 库路径，
# 确保 onnxruntime-gpu 1.14.1 能找到兼容的 CUDA 11.x 运行时库。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

# CUDA 11.8 库路径（onnxruntime-gpu 1.14.1 基于 CUDA 11.6 编译）
CUDA11_LIB="/usr/local/cuda-11.8/lib64"
if [ -d "$CUDA11_LIB" ]; then
    export LD_LIBRARY_PATH="$CUDA11_LIB:${LD_LIBRARY_PATH:-}"
    # 预加载 CUDA 11.8 的 cuBLAS/cuBLASLt，避免系统 CUDA 12.x 版本被加载
    # 这两个库同时加载会多占 ~900 MB 内存
    PRELOAD_LIBS=""
    for lib in libcublas.so.11 libcublasLt.so.11; do
        path="$CUDA11_LIB/$lib"
        [ -f "$path" ] && PRELOAD_LIBS="$path:$PRELOAD_LIBS"
    done
    [ -n "$PRELOAD_LIBS" ] && export LD_PRELOAD="${PRELOAD_LIBS}${LD_PRELOAD:-}"
fi

exec "$VENV_DIR/bin/python3" "$SCRIPT_DIR/ibus/main.py" "$@"
