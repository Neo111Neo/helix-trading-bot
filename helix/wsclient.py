"""Minimal WebSocket client (RFC 6455) using only the standard library.

Enough for market-data streams: TLS, handshake verification, masked client frames,
fragmented messages, ping/pong and close. A socket timeout never loses buffered bytes,
so `recv()` can be called with a short timeout in a loop."""

from __future__ import annotations

import base64
import hashlib
import os
import socket
import ssl
import struct
import threading
from urllib.parse import urlparse

_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
OP_CONT, OP_TEXT, OP_BINARY, OP_CLOSE, OP_PING, OP_PONG = 0x0, 0x1, 0x2, 0x8, 0x9, 0xA


class WebSocketError(Exception):
    pass


class WebSocketClosed(WebSocketError):
    pass


def _mask(data: bytes, key: bytes) -> bytes:
    if not data:
        return data
    n = len(data)
    k = (key * (n // 4 + 1))[:n]
    return (int.from_bytes(data, "big") ^ int.from_bytes(k, "big")).to_bytes(n, "big")


def accept_key(key: str) -> str:
    return base64.b64encode(hashlib.sha1((key + _GUID).encode()).digest()).decode()


class WebSocket:
    def __init__(self, url: str, timeout: float = 10.0, headers: dict | None = None,
                 ssl_context: ssl.SSLContext | None = None):
        u = urlparse(url)
        if u.scheme not in ("ws", "wss") or not u.hostname:
            raise ValueError(f"Not a WebSocket URL: {url}")
        self.url = url
        self.host = u.hostname
        self.port = u.port or (443 if u.scheme == "wss" else 80)
        self.path = (u.path or "/") + (f"?{u.query}" if u.query else "")
        sock = socket.create_connection((self.host, self.port), timeout=timeout)
        if u.scheme == "wss":
            ctx = ssl_context or ssl.create_default_context()
            try:
                sock = ctx.wrap_socket(sock, server_hostname=self.host)
            except Exception:
                sock.close()
                raise
        self.sock = sock
        self._buf = bytearray()
        self._frag = bytearray()
        self._frag_op: int | None = None
        self._send_lock = threading.Lock()
        self.closed = False
        try:
            self._handshake(headers or {})
        except Exception:
            self.sock.close()
            raise

    # ---- handshake -------------------------------------------------------------------
    def _handshake(self, headers: dict) -> None:
        key = base64.b64encode(os.urandom(16)).decode()
        host = self.host if self.port in (80, 443) else f"{self.host}:{self.port}"
        lines = [f"GET {self.path} HTTP/1.1", f"Host: {host}", "Upgrade: websocket", "Connection: Upgrade",
                 f"Sec-WebSocket-Key: {key}", "Sec-WebSocket-Version: 13"]
        lines += [f"{k}: {v}" for k, v in headers.items()]
        self.sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1"))
        while b"\r\n\r\n" not in self._buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise WebSocketClosed("Connection closed during handshake")
            self._buf += chunk
            if len(self._buf) > 65536:
                raise WebSocketError("Handshake response too large")
        end = self._buf.index(b"\r\n\r\n")
        head = bytes(self._buf[:end]).decode("latin-1")
        del self._buf[: end + 4]
        status = head.split("\r\n", 1)[0]
        parts = status.split(" ")
        if len(parts) < 2 or parts[1] != "101":
            raise WebSocketError(f"Handshake failed: {status}")
        hdrs = {}
        for line in head.split("\r\n")[1:]:
            if ":" in line:
                k, v = line.split(":", 1)
                hdrs[k.strip().lower()] = v.strip()
        if hdrs.get("sec-websocket-accept") != accept_key(key):
            raise WebSocketError("Server sent a bad Sec-WebSocket-Accept header")

    # ---- sending ---------------------------------------------------------------------
    def _send_frame(self, opcode: int, payload: bytes) -> None:
        n = len(payload)
        head = bytearray([0x80 | opcode])
        if n < 126:
            head.append(0x80 | n)
        elif n < 65536:
            head.append(0x80 | 126)
            head += struct.pack("!H", n)
        else:
            head.append(0x80 | 127)
            head += struct.pack("!Q", n)
        key = os.urandom(4)
        with self._send_lock:
            self.sock.sendall(bytes(head) + key + _mask(payload, key))

    def send(self, text: str) -> None:
        if self.closed:
            raise WebSocketClosed("Socket is closed")
        self._send_frame(OP_TEXT, text.encode("utf-8"))

    def settimeout(self, seconds: float | None) -> None:
        self.sock.settimeout(seconds)

    # ---- receiving -------------------------------------------------------------------
    def _parse_frame(self):
        buf = self._buf
        if len(buf) < 2:
            return None
        b1, b2 = buf[0], buf[1]
        n, i = b2 & 0x7F, 2
        if n == 126:
            if len(buf) < 4:
                return None
            n, i = struct.unpack_from("!H", buf, 2)[0], 4
        elif n == 127:
            if len(buf) < 10:
                return None
            n, i = struct.unpack_from("!Q", buf, 2)[0], 10
        key = None
        if b2 & 0x80:
            if len(buf) < i + 4:
                return None
            key, i = bytes(buf[i:i + 4]), i + 4
        if len(buf) < i + n:
            return None
        payload = bytes(buf[i:i + n])
        del buf[: i + n]
        if key:
            payload = _mask(payload, key)
        return bool(b1 & 0x80), b1 & 0x0F, payload

    def recv(self) -> str:
        """Next complete text message. Raises socket.timeout (buffer kept) or WebSocketClosed."""
        while True:
            frame = self._parse_frame()
            if frame is None:
                if self.closed:
                    raise WebSocketClosed("Socket is closed")
                chunk = self.sock.recv(65536)
                if not chunk:
                    self.closed = True
                    self._drop()
                    raise WebSocketClosed("Connection closed by server")
                self._buf += chunk
                continue
            fin, op, payload = frame
            if op == OP_PING:
                self._send_frame(OP_PONG, payload)
                continue
            if op == OP_PONG:
                continue
            if op == OP_CLOSE:
                code = struct.unpack("!H", payload[:2])[0] if len(payload) >= 2 else 1005
                self.closed = True
                try:
                    self._send_frame(OP_CLOSE, payload[:2])
                except OSError:
                    pass
                self._drop()
                raise WebSocketClosed(f"Closed by server (code {code})")
            if op in (OP_TEXT, OP_BINARY):
                self._frag_op, self._frag = op, bytearray(payload)
            elif op == OP_CONT and self._frag_op is not None:
                self._frag += payload
            else:
                raise WebSocketError(f"Unexpected opcode {op}")
            if fin:
                data, self._frag_op, self._frag = bytes(self._frag), None, bytearray()
                return data.decode("utf-8", errors="replace")

    def _drop(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            try:
                self._send_frame(OP_CLOSE, struct.pack("!H", 1000))
            except OSError:
                pass
        try:
            self.sock.close()
        except OSError:
            pass
