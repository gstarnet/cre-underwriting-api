from __future__ import annotations

from src.whatif import run_whatif


class _DummyModel:
    def predict(self, X):  # noqa: N803 - sklearn-style signature
        return [1_000_000.0] * len(X)


def test_run_whatif_uses_scalar_exit_cap_as_default():
    results = run_whatif(
        model=_DummyModel(),
        base_features={
            "purchase_price": 25_000_000.0,
            "ltv": 0.65,
            "interest_rate": 0.062,
            "amort_years": 25,
        },
        amort_years=25,
        base_exit_cap_rate=0.08,
        max_scenarios=1,
    )

    assert len(results) == 1
    assert results[0].exit_cap_rate == 0.08
