#!/bin/bash
# 安装 IBus 组件文件
# 用法: sudo scripts/install_ibus_component.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPONENT_SRC="$PROJECT_DIR/ibus/vocotype.xml"
COMPONENT_DST="/usr/share/ibus/component/vocotype.xml"

if [ "$(id -u)" -ne 0 ]; then
    echo "需要 root 权限，请使用 sudo 运行" >&2
    exit 1
fi

if [ ! -f "$COMPONENT_SRC" ]; then
    echo "找不到组件模板: $COMPONENT_SRC" >&2
    exit 1
fi

# 备份旧文件
if [ -f "$COMPONENT_DST" ]; then
    cp "$COMPONENT_DST" "${COMPONENT_DST}.bak.$(date +%Y%m%d%H%M%S)"
    echo "已备份旧组件文件"
fi

# 替换 @PROJECT_DIR@ 占位符并安装
sed "s|@PROJECT_DIR@|$PROJECT_DIR|g" "$COMPONENT_SRC" > "$COMPONENT_DST"
chmod 644 "$COMPONENT_DST"

echo "已安装组件文件: $COMPONENT_DST"
echo "启动命令: $(grep '<exec>' "$COMPONENT_DST" | sed 's/.*<exec>\(.*\)<\/exec>.*/\1/')"

# 重启 IBus
if command -v ibus >/dev/null 2>&1; then
    echo "重启 IBus..."
    ibus restart 2>/dev/null || true
fi
