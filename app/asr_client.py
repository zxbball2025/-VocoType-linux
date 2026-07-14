#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ASR 守护进程客户端 — 接口兼容 FunASRServer 的鸭子类型替身。

RemoteASR 提供与 FunASRServer 相同的方法签名
（initialize / transcribe_audio / cleanup / device / transcription_count），
调用方无需感知模型在本进程还是守护进程。

用法：
    from app.asr_client import RemoteASR, daemon_available

    if daemon_available():
        server = RemoteASR()
    else:
        server = FunASRServer(device="cuda:0")   # 回退本地加载
"""
from __future__ import annotations

import json
import logging
import os
import socket

logger = logging.getLogger(__name__)


def default_socket_path() -> str:
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR", "/tmp")
    return os.path.join(runtime_dir, "vocotype-asr.sock")


SOCKET_PATH = os.environ.get("VOCOTYPE_ASR_SOCKET", default_socket_path())


def _request(payload: dict, timeout: float) -> dict:
    """发一个 JSON 请求，读到 EOF，返回 JSON 响应。"""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect(SOCKET_PATH)
        sock.sendall(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        sock.shutdown(socket.SHUT_WR)
        chunks = []
        while True:
            chunk = sock.recv(8192)
            if not chunk:
                break
            chunks.append(chunk)
    if not chunks:
        raise ConnectionError("ASR daemon 空响应")
    return json.loads(b"".join(chunks).decode("utf-8"))


def daemon_available(timeout: float = 2.0) -> bool:
    """守护进程是否存活可用（ping）。"""
    try:
        resp = _request({"type": "ping"}, timeout=timeout)
        return bool(resp.get("pong"))
    except Exception:
        return False


class RemoteASR:
    """通过 Unix socket 调用 asr_daemon 的 FunASRServer 替身。"""

    def __init__(self, device: str | None = None, timeout: float = 60.0):
        # device 参数仅为接口兼容；实际设备由 daemon 决定
        self.timeout = timeout
        self.initialized = False
        self.device = device or "remote"
        self.transcription_count = 0

    def initialize(self) -> dict:
        try:
            resp = _request({"type": "ping"}, timeout=5.0)
        except Exception as exc:
            return {
                "success": False,
                "error": f"ASR daemon 不可达 ({SOCKET_PATH}): {exc}",
                "type": "daemon_error",
            }
        if not resp.get("pong"):
            return {"success": False, "error": "ASR daemon ping 异常", "type": "daemon_error"}
        self.device = resp.get("device", "remote")
        self.initialized = True
        logger.info("已连接 ASR daemon（设备=%s，socket=%s）", self.device, SOCKET_PATH)
        return {"success": True, "message": f"connected to ASR daemon ({self.device})"}

    def transcribe_audio(self, audio_path, options=None) -> dict:
        if not self.initialized:
            init_result = self.initialize()
            if not init_result.get("success"):
                return init_result
        try:
            result = _request(
                {"type": "transcribe", "audio_path": str(audio_path), "options": options},
                timeout=self.timeout,
            )
        except Exception as exc:
            logger.error("ASR daemon 请求失败: %s", exc)
            self.initialized = False  # 下次调用重新探测
            return {"success": False, "error": f"ASR daemon 请求失败: {exc}", "type": "daemon_error"}
        if result.get("success"):
            self.transcription_count += 1
        return result

    def cleanup(self) -> None:
        # 模型属于 daemon，客户端无资源可清
        self.initialized = False
