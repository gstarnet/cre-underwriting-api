from __future__ import annotations

from types import SimpleNamespace

from src import mcp_service


def _request(method: str, params=None, request_id=1):
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params or {},
    }


def _predict_like_payload() -> dict:
    return {
        "deal_id": "D1",
        "asof_date": "2024-06-01",
        "property_type": "Industrial",
        "city": "Tampa",
        "state": "FL",
        "zip": "33602",
        "year_built": 2008,
        "gross_leasable_sqft": 125000,
        "units": None,
        "purchase_price": 25000000,
        "noi_t12": 1500000,
        "occupancy_t12": 0.95,
        "opex_t12": 450000,
        "gross_rent_t12": 2200000,
        "ltv": 0.65,
        "interest_rate": 0.062,
        "amort_years": 25,
        "exit_cap_rate": 0.065,
        "selling_cost_pct": 0.02,
    }


def test_initialize_and_tools_list_exposes_institutional_tools():
    init_resp = mcp_service.handle_request(_request("initialize"))
    assert init_resp["result"]["serverInfo"]["name"] == "cre-underwriting-mcp"

    list_resp = mcp_service.handle_request(_request("tools/list"))
    names = {t["name"] for t in list_resp["result"]["tools"]}
    assert "underwrite_inst" in names
    assert "whatif_inst" in names


def test_tools_call_underwrite_inst_routes_success(monkeypatch):
    def _fake_underwrite_inst_endpoint(_req):
        return SimpleNamespace(
            model_dump=lambda: {
                "predicted_noi_next12": 123456.0,
                "institutional_underwriting": {"summary": {"irr": 0.12}},
            }
        )

    monkeypatch.setattr(
        mcp_service.api,
        "underwrite_inst_endpoint",
        _fake_underwrite_inst_endpoint,
    )

    req = _request(
        "tools/call",
        params={
            "name": "underwrite_inst",
            "arguments": _predict_like_payload(),
        },
    )
    resp = mcp_service.handle_request(req)
    assert resp["result"]["isError"] is False
    out = resp["result"]["structuredContent"]
    assert out["predicted_noi_next12"] == 123456.0
    assert "institutional_underwriting" in out


def test_tools_call_whatif_inst_routes_success(monkeypatch):
    def _fake_whatif_inst(_req):
        return SimpleNamespace(
            model_dump=lambda: {
                "scenarios": [
                    {
                        "inputs": {"purchase_price": 25000000},
                        "predicted_noi_next12": 2000000.0,
                        "summary": {"irr": 0.11},
                    }
                ]
            }
        )

    monkeypatch.setattr(mcp_service.api, "whatif_inst", _fake_whatif_inst)

    payload = _predict_like_payload()
    payload.update(
        {
            "purchase_prices": [24000000, 25000000],
            "exit_cap_rates": [0.0625, 0.0675],
            "max_scenarios": 20,
            "top_n": 5,
            "sort_by": "irr",
        }
    )
    req = _request(
        "tools/call",
        params={"name": "whatif_inst", "arguments": payload},
    )
    resp = mcp_service.handle_request(req)
    assert resp["result"]["isError"] is False
    assert len(resp["result"]["structuredContent"]["scenarios"]) == 1


def test_tools_call_unknown_tool_returns_error():
    req = _request("tools/call", params={"name": "does_not_exist", "arguments": {}})
    resp = mcp_service.handle_request(req)
    assert resp["error"]["code"] == -32602
