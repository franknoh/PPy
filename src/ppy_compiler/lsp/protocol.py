"""Minimal LSP base protocol: framed JSON-RPC over a byte stream."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, BinaryIO, Iterator

__all__ = ["Message", "read_messages", "write_message", "encode"]


@dataclass(frozen=True, slots=True)
class Message:
    payload: dict[str, Any]

    @property
    def method(self) -> str | None:
        return self.payload.get("method")

    @property
    def id(self) -> object:
        return self.payload.get("id")

    @property
    def params(self) -> dict[str, Any]:
        found = self.payload.get("params")
        return found if isinstance(found, dict) else {}

    @property
    def is_request(self) -> bool:
        return "id" in self.payload and "method" in self.payload


def _read_headers(stream: BinaryIO) -> dict[str, str]:
    headers: dict[str, str] = {}
    while True:
        line = stream.readline()
        if not line or line in (b"\r\n", b"\n"):
            return headers
        name, _, value = line.decode("ascii", "replace").partition(":")
        headers[name.strip().lower()] = value.strip()


def read_messages(stream: BinaryIO) -> Iterator[Message]:
    """Yield each framed JSON-RPC message until the stream closes."""
    while True:
        headers = _read_headers(stream)
        if not headers:
            return
        try:
            length = int(headers.get("content-length", "0"))
        except ValueError:
            return
        if length <= 0:
            continue
        body = stream.read(length)
        if not body or len(body) < length:
            return
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            yield Message(payload)


def encode(payload: dict[str, Any]) -> bytes:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return b"Content-Length: " + str(len(body)).encode("ascii") + b"\r\n\r\n" + body


def write_message(stream: BinaryIO, payload: dict[str, Any]) -> None:
    stream.write(encode(payload))
    stream.flush()
