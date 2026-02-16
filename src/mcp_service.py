from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from pydantic import ValidationError

from src import api

MCP_SERVER_NAME = "cre-underwriting-mcp"
MCP_SERVER_VERSION = "0.1.0"
MCP_PROTOCOL_VERSION = "2024-11-05"


def _tool_specs() -> List[Dict[str, Any]]:
    return [
        {
            "name": "health",
            "description": "Return API health status.",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
        {
            "name": "explainability",
            "description": "Return explainability payload with traceability metadata.",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
        {
            "name": "predict",
            "description": "Predict next-12 NOI and return ROI snapshot for a CRE deal.",
            "inputSchema": api.PredictRequest.model_json_schema(),
        },
        {
            "name": "predict_features",
            "description": "Predict next-12 NOI from free-form features payload.",
            "inputSchema": api.PredictFeaturesRequest.model_json_schema(),
        },
        {
            "name": "whatif",
            "description": "Run baseline what-if scenario grid.",
            "inputSchema": api.WhatIfRequest.model_json_schema(),
        },
        {
            "name": "underwrite",
            "description": "Run baseline multi-year underwriting.",
            "inputSchema": api.UnderwriteRequest.model_json_schema(),
        },
        {
            "name": "underwrite_inst",
            "description": "Run institutional underwriting (v2).",
            "inputSchema": api.UnderwriteInstRequest.model_json_schema(),
        },
        {
            "name": "whatif_inst",
            "description": "Run institutional what-if scenario grid with guardrails.",
            "inputSchema": api.WhatIfInstRequest.model_json_schema(),
        },
    ]


def _call_tool(name: str, args: Optional[Dict[str, Any]]) -> Any:
    payload = args or {}

    if name == "health":
        return api.health()
    if name == "explainability":
        return api.explainability()
    if name == "predict":
        return api.predict(api.PredictRequest(**payload)).model_dump()
    if name == "predict_features":
        return api.predict_features(api.PredictFeaturesRequest(**payload)).model_dump()
    if name == "whatif":
        return api.whatif(api.WhatIfRequest(**payload)).model_dump()
    if name == "underwrite":
        return api.underwrite_endpoint(api.UnderwriteRequest(**payload)).model_dump()
    if name == "underwrite_inst":
        return api.underwrite_inst_endpoint(api.UnderwriteInstRequest(**payload)).model_dump()
    if name == "whatif_inst":
        return api.whatif_inst(api.WhatIfInstRequest(**payload)).model_dump()

    raise ValueError(f"Unknown tool: {name}")


def _response(result: Any, *, request_id: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error_response(
    code: int,
    message: str,
    *,
    request_id: Any = None,
    data: Any = None,
) -> Dict[str, Any]:
    err: Dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": err}


def handle_request(request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if request.get("jsonrpc") != "2.0":
        return _error_response(-32600, "Invalid Request", request_id=request.get("id"))

    method = request.get("method")
    params = request.get("params", {})
    request_id = request.get("id")

    # Notifications have no response.
    if request_id is None:
        return None

    try:
        if method == "initialize":
            return _response(
                {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": MCP_SERVER_NAME,
                        "version": MCP_SERVER_VERSION,
                    },
                },
                request_id=request_id,
            )

        if method == "ping":
            return _response({}, request_id=request_id)

        if method == "tools/list":
            return _response({"tools": _tool_specs()}, request_id=request_id)

        if method == "tools/call":
            name = params.get("name")
            if not isinstance(name, str) or not name:
                return _error_response(
                    -32602,
                    "Invalid params",
                    request_id=request_id,
                    data="tools/call requires non-empty string `name`",
                )
            args = params.get("arguments", {})
            if args is None:
                args = {}
            if not isinstance(args, dict):
                return _error_response(
                    -32602,
                    "Invalid params",
                    request_id=request_id,
                    data="tools/call `arguments` must be an object",
                )

            out = _call_tool(name, args)
            return _response(
                {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(out, ensure_ascii=True),
                        }
                    ],
                    "structuredContent": out,
                    "isError": False,
                },
                request_id=request_id,
            )

        return _error_response(-32601, "Method not found", request_id=request_id)
    except ValidationError as e:
        return _error_response(
            -32602,
            "Invalid params",
            request_id=request_id,
            data=e.errors(),
        )
    except HTTPException as e:
        return _response(
            {
                "content": [{"type": "text", "text": str(e.detail)}],
                "structuredContent": {"status_code": e.status_code, "detail": e.detail},
                "isError": True,
            },
            request_id=request_id,
        )
    except ValueError as e:
        return _error_response(-32602, "Invalid params", request_id=request_id, data=str(e))
    except Exception as e:  # pragma: no cover
        return _error_response(-32603, "Internal error", request_id=request_id, data=str(e))


def _read_message() -> Optional[Dict[str, Any]]:
    headers: Dict[str, str] = {}

    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break

        decoded = line.decode("utf-8").strip()
        if ":" not in decoded:
            continue
        k, v = decoded.split(":", 1)
        headers[k.strip().lower()] = v.strip()

    if "content-length" not in headers:
        return None

    length = int(headers["content-length"])
    body = sys.stdin.buffer.read(length)
    if not body:
        return None
    return json.loads(body.decode("utf-8"))


def _write_message(payload: Dict[str, Any]) -> None:
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(data)}\r\n\r\n".encode("ascii"))
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


def main() -> None:
    while True:
        message = _read_message()
        if message is None:
            break

        if isinstance(message, list):
            responses = [handle_request(item) for item in message]
            out = [r for r in responses if r is not None]
            if out:
                _write_message(out)
            continue

        if not isinstance(message, dict):
            _write_message(_error_response(-32600, "Invalid Request", request_id=None))
            continue

        response = handle_request(message)
        if response is not None:
            _write_message(response)


if __name__ == "__main__":
    main()
