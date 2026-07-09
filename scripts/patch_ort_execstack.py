#!/usr/bin/env python3
"""修补 onnxruntime-gpu 1.14.1 的 ELF GNU_STACK 段，去掉可执行标记。

解决现代 Linux 内核 (6.x+) 拒绝加载 executable stack 共享对象的问题：
  ImportError: cannot enable executable stack as shared object requires: Invalid argument

用法：
  python3 scripts/patch_ort_execstack.py
  # 或指定 venv 路径：
  python3 scripts/patch_ort_execstack.py /path/to/venv
"""
import glob
import struct
import sys
from pathlib import Path


def find_ort_so(venv_path: str | None = None) -> str | None:
    if venv_path:
        pattern = str(Path(venv_path) / "lib/python*/site-packages/onnxruntime/capi/onnxruntime_pybind11_state*.so")
    else:
        pattern = str(Path(__file__).resolve().parent.parent / ".venv/lib/python*/site-packages/onnxruntime/capi/onnxruntime_pybind11_state*.so")
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def patch_execstack(so_path: str) -> bool:
    with open(so_path, "rb") as f:
        data = bytearray(f.read())

    if data[:4] != b"\x7fELF":
        print(f"不是 ELF 文件: {so_path}")
        return False

    is_64 = data[4] == 2
    if not is_64:
        print("仅支持 64 位 ELF")
        return False

    e_phoff = struct.unpack_from("<Q", data, 32)[0]
    e_phentsize = struct.unpack_from("<H", data, 54)[0]
    e_phnum = struct.unpack_from("<H", data, 56)[0]

    PT_GNU_STACK = 0x6474E551
    PF_X = 1

    for i in range(e_phnum):
        off = e_phoff + i * e_phentsize
        p_type = struct.unpack_from("<I", data, off)[0]
        if p_type == PT_GNU_STACK:
            p_flags_off = off + 4
            old_flags = struct.unpack_from("<I", data, p_flags_off)[0]
            if old_flags & PF_X:
                new_flags = old_flags & ~PF_X
                struct.pack_into("<I", data, p_flags_off, new_flags)
                with open(so_path, "wb") as f:
                    f.write(data)
                print(f"已修补: {so_path}")
                print(f"  GNU_STACK flags: 0x{old_flags:x} (RWE) -> 0x{new_flags:x} (RW)")
                return True
            else:
                print(f"已修补过 (flags=0x{old_flags:x})，无需操作")
                return True

    print("未找到 PT_GNU_STACK 段")
    return False


def main():
    venv = sys.argv[1] if len(sys.argv) > 1 else None
    so = find_ort_so(venv)
    if not so:
        print("找不到 onnxruntime_pybind11_state*.so")
        sys.exit(1)
    print(f"目标: {so}")
    ok = patch_execstack(so)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
