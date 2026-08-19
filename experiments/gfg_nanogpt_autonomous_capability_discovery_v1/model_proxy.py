from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import select
import socket
import socketserver
import threading
from typing import Any


MAX_HEADER_BYTES = 16 * 1024
HEADER_TERMINATOR = b"\r\n\r\n"


def _split_header(data: bytes) -> tuple[bytes, bytes]:
    marker = data.find(HEADER_TERMINATOR)
    if marker < 0:
        raise ValueError("PROXY_HEADER_INCOMPLETE")
    end = marker + len(HEADER_TERMINATOR)
    return data[:end], data[end:]


def _connect_target(header: bytes) -> tuple[str, int] | None:
    try:
        first = header.split(b"\r\n", 1)[0].decode("ascii")
        method, authority, _version = first.split(" ", 2)
        host, port_text = authority.rsplit(":", 1)
        port = int(port_text)
    except (UnicodeDecodeError, ValueError):
        return None
    if method != "CONNECT" or not host or not 1 <= port <= 65535:
        return None
    return host.lower(), port


class AllowlistProxy(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        upstream: tuple[str, int],
        allowed_hosts: set[str],
        audit_log: Path,
    ) -> None:
        super().__init__(server_address, ConnectHandler)
        self.upstream = upstream
        self.allowed_hosts = allowed_hosts
        self.audit_log = audit_log
        self.audit_lock = threading.Lock()

    def audit(self, row: dict[str, Any]) -> None:
        material = {
            "at": datetime.now(timezone.utc).isoformat(),
            **row,
        }
        with self.audit_lock:
            with self.audit_log.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        material,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )


class ConnectHandler(socketserver.BaseRequestHandler):
    server: AllowlistProxy

    def _header(self) -> tuple[bytes, bytes]:
        data = bytearray()
        self.request.settimeout(30)
        while HEADER_TERMINATOR not in data:
            chunk = self.request.recv(4096)
            if not chunk:
                break
            data.extend(chunk)
            if len(data) > MAX_HEADER_BYTES:
                raise ValueError("PROXY_HEADER_TOO_LARGE")
        return _split_header(bytes(data))

    def handle(self) -> None:
        target: tuple[str, int] | None = None
        try:
            header, client_prefetch = self._header()
            if header.startswith(b"GET /health HTTP/1.1\r\n"):
                self.request.sendall(
                    b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK"
                )
                return
            target = _connect_target(header)
            if (
                target is None
                or target[0] not in self.server.allowed_hosts
                or target[1] != 443
            ):
                self.request.sendall(
                    b"HTTP/1.1 403 Forbidden\r\nConnection: close\r\n\r\n"
                )
                self.server.audit(
                    {
                        "allowed": False,
                        "host": target[0] if target else None,
                        "port": target[1] if target else None,
                        "status": "DENIED",
                    }
                )
                return
            upstream = socket.create_connection(self.server.upstream, 30)
            try:
                upstream.sendall(header)
                response = bytearray()
                while HEADER_TERMINATOR not in response:
                    chunk = upstream.recv(4096)
                    if not chunk:
                        break
                    response.extend(chunk)
                    if len(response) > MAX_HEADER_BYTES:
                        raise ValueError("UPSTREAM_HEADER_TOO_LARGE")
                response_header, upstream_prefetch = _split_header(
                    bytes(response)
                )
                self.request.sendall(response_header)
                first = response_header.split(b"\r\n", 1)[0]
                if b" 200 " not in first:
                    raise ConnectionError("UPSTREAM_CONNECT_REJECTED")
                if upstream_prefetch:
                    self.request.sendall(upstream_prefetch)
                if client_prefetch:
                    upstream.sendall(client_prefetch)
                self.server.audit(
                    {
                        "allowed": True,
                        "host": target[0],
                        "port": target[1],
                        "status": "TUNNEL_ESTABLISHED",
                    }
                )
                self.request.settimeout(None)
                upstream.settimeout(None)
                peers = [self.request, upstream]
                while peers:
                    readable, _, exceptional = select.select(
                        peers, [], peers, 60
                    )
                    if exceptional:
                        break
                    if not readable:
                        continue
                    for source in readable:
                        data = source.recv(65536)
                        if not data:
                            return
                        destination = (
                            upstream
                            if source is self.request
                            else self.request
                        )
                        destination.sendall(data)
            finally:
                upstream.close()
        except Exception as exc:
            self.server.audit(
                {
                    "allowed": bool(
                        target
                        and target[0] in self.server.allowed_hosts
                        and target[1] == 443
                    ),
                    "error": type(exc).__name__,
                    "host": target[0] if target else None,
                    "port": target[1] if target else None,
                    "status": "ERROR",
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--upstream-host", required=True)
    parser.add_argument("--upstream-port", type=int, required=True)
    parser.add_argument("--allowed-host", action="append", required=True)
    parser.add_argument("--audit-log", type=Path, required=True)
    args = parser.parse_args()
    args.audit_log.parent.mkdir(parents=True, exist_ok=True)
    proxy = AllowlistProxy(
        (args.listen_host, args.listen_port),
        upstream=(args.upstream_host, args.upstream_port),
        allowed_hosts={host.lower() for host in args.allowed_host},
        audit_log=args.audit_log,
    )
    proxy.serve_forever(poll_interval=0.25)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
