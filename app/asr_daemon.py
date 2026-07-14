#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立 ASR 守护进程 — 全机唯一一份 FunASR 模型，多客户端共享。

借用 fcitx5/backend/fcitx5_server.py 的 socket 框架，砍掉 Rime/SLM，
与任何输入法框架无关。IBus 输入法和 m5stack3 语音助手都通过
Unix socket 调用本进程，GPU 显存和 CPU 内存只占一份。

IPC 协议（JSON，每连接一个请求，读到 EOF 为止）：
  {"type": "transcribe", "audio_path": "/tmp/xxx.wav", "options": {...}}
    -> {"success": true, "text": "识别结果", "raw_text": "...", ...}
  {"type": "ping"}
    -> {"pong": true, "device": "cuda:0", "count": 42}

启动：
  /home/zxb7559/VocoType-linux/.venv/bin/python3 -m app.asr_daemon
或走 systemd 用户服务 vocotype-asr.service。
"""
from __future__ import annotations

import json
import logging
import os
import signal
import socket
import stat
import sys
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.funasr_server import FunASRServer
from app.logging_config import setup_logging

logger = logging.getLogger(__name__)


def default_socket_path() -> str:
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR", "/tmp")
    return os.path.join(runtime_dir, "vocotype-asr.sock")


SOCKET_PATH = os.environ.get("VOCOTYPE_ASR_SOCKET", default_socket_path())
MAX_REQUEST_BYTES = 1024 * 1024
REQUEST_TIMEOUT_S = 60.0  # 长音频推理可能数十秒


class ASRDaemon:
    """纯 ASR socket 服务：transcribe + ping。"""

    def __init__(self, device: str | None = None):
        logger.info("正在初始化 FunASR（守护进程，唯一实例）...")
        self.asr_server = FunASRServer(device=device or "cuda:0")
        result = self.asr_server.initialize()
        if not result.get("success"):
            logger.error("FunASR 初始化失败: %s", result.get("error"))
            sys.exit(1)
        logger.info("FunASR 初始化成功，设备=%s", self.asr_server.device)

        self.running = True
        # 注：FunASRServer 的 ONNX Runtime InferenceSession 对并发推理是线程安全的，
        # 不需要全局锁。如果多个客户端同时请求，ONNX Runtime 会在 GPU 上排队执行。

        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        logger.info("收到信号 %d，准备退出...", signum)
        self.running = False

    def _cleanup_socket_path(self, path: str) -> None:
        if not os.path.exists(path):
            return
        st = os.lstat(path)
        if stat.S_ISSOCK(st.st_mode) or stat.S_ISLNK(st.st_mode):
            os.remove(path)
            logger.info("已移除旧 socket: %s", path)
        else:
            raise RuntimeError(f"socket 路径已存在且不是 socket: {path}")

    def run(self):
        self._cleanup_socket_path(SOCKET_PATH)

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(SOCKET_PATH)
        os.chmod(SOCKET_PATH, 0o600)
        sock.listen(8)
        sock.settimeout(1.0)

        logger.info("ASR 守护进程已启动，监听: %s", SOCKET_PATH)

        try:
            while self.running:
                try:
                    conn, _ = sock.accept()
                    threading.Thread(
                        target=self.handle_client,
                        args=(conn,),
                        daemon=True,
                        name="ASRDaemonClient",
                    ).start()
                except socket.timeout:
                    continue
                except Exception as exc:
                    if self.running:
                        logger.error("接受连接失败: %s", exc)
        finally:
            sock.close()
            try:
                self._cleanup_socket_path(SOCKET_PATH)
            except RuntimeError as exc:
                logger.warning("清理 socket 失败: %s", exc)
            self.asr_server.cleanup()
            logger.info("ASR 守护进程已停止")

    def handle_client(self, conn: socket.socket):
        try:
            conn.settimeout(REQUEST_TIMEOUT_S)
            chunks = []
            total = 0
            while True:
                chunk = conn.recv(8192)
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > MAX_REQUEST_BYTES:
                    self._send(conn, {"success": False, "error": "request too large"})
                    return
            if not chunks:
                return

            try:
                request = json.loads(b"".join(chunks).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                self._send(conn, {"success": False, "error": f"bad request: {exc}"})
                return

            req_type = request.get("type")

            if req_type == "ping":
                self._send(conn, {
                    "pong": True,
                    "device": str(self.asr_server.device),
                    "count": self.asr_server.transcription_count,
                })
            elif req_type == "transcribe":
                audio_path = request.get("audio_path")
                if not audio_path:
                    self._send(conn, {"success": False, "error": "缺少 audio_path"})
                    return
                options = request.get("options") or None
                result = self.asr_server.transcribe_audio(audio_path, options=options)
                self._send(conn, result)
            else:
                self._send(conn, {"success": False, "error": f"unknown type: {req_type}"})
        except Exception as exc:
            logger.error("处理请求失败: %s", exc, exc_info=True)
            try:
                self._send(conn, {"success": False, "error": str(exc)})
            except Exception:
                pass
        finally:
            try:
                conn.close()
            except Exception:
                pass

    @staticmethod
    def _send(conn: socket.socket, obj: dict) -> None:
        conn.sendall(json.dumps(obj, ensure_ascii=False).encode("utf-8"))
        # 半关闭写端，让客户端 recv 到 EOF
        try:
            conn.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def main():
    log_dir = os.path.join(str(PROJECT_ROOT), "logs")
    setup_logging("INFO", log_dir)

    device = os.environ.get("VOCOTYPE_ASR_DEVICE")  # 默认 cuda:0，可覆盖
    daemon = ASRDaemon(device=device)
    daemon.run()


if __name__ == "__main__":
    main()
