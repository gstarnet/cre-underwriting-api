#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


def _send_rpc(proc: subprocess.Popen, payload: Dict[str, Any]) -> Dict[str, Any]:
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    msg = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body
    assert proc.stdin is not None
    assert proc.stdout is not None
    proc.stdin.write(msg)
    proc.stdin.flush()

    header = b""
    while b"\r\n\r\n" not in header:
        ch = proc.stdout.read(1)
        if not ch:
            raise RuntimeError("EOF while reading MCP response headers")
        header += ch

    header_text = header.decode("utf-8", errors="replace")
    length = None
    for line in header_text.split("\r\n"):
        if line.lower().startswith("content-length:"):
            length = int(line.split(":", 1)[1].strip())
            break
    if length is None:
        raise RuntimeError("Missing Content-Length in MCP response")

    body_bytes = proc.stdout.read(length)
    if len(body_bytes) != length:
        raise RuntimeError("Incomplete MCP response body")
    return json.loads(body_bytes.decode("utf-8"))


def _expect_ok(resp: Dict[str, Any], label: str) -> Dict[str, Any]:
    if "error" in resp:
        raise RuntimeError(f"{label} failed: {resp['error']}")
    return resp["result"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Quick MCP stdio connectivity test client.")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use for launching src.mcp_service (default: current interpreter).",
    )
    parser.add_argument(
        "--cwd",
        default=str(Path(__file__).resolve().parents[1]),
        help="Project root containing src/ (default: repo root inferred from this script).",
    )
    args = parser.parse_args()

    proc = subprocess.Popen(
        [args.python, "-m", "src.mcp_service"],
        cwd=args.cwd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        init = _expect_ok(
            _send_rpc(
                proc,
                {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            ),
            "initialize",
        )
        print(f"OK initialize: server={init['serverInfo']['name']} version={init['serverInfo']['version']}")

        tools = _expect_ok(
            _send_rpc(
                proc,
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            ),
            "tools/list",
        )
        names = sorted(t["name"] for t in tools.get("tools", []))
        print(f"OK tools/list: count={len(names)} tools={names}")

        call = _expect_ok(
            _send_rpc(
                proc,
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {"name": "health", "arguments": {}},
                },
            ),
            "tools/call(health)",
        )
        status = call.get("structuredContent", {}).get("status")
        if status != "ok":
            raise RuntimeError(f"Unexpected health status: {status!r}")
        print("OK tools/call health")
        print("MCP connectivity test passed.")
        return 0
    except Exception as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
