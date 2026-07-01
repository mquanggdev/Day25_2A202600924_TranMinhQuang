"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num
from finops import pricing, sustainability

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    base_cost = opt_cost = 0.0
    total_tokens = 0

    reasoning_count = 0
    reasoning_tokens = 0
    reasoning_base_cost = 0.0
    reasoning_opt_cost = 0.0
    reasoning_energy_wh = 0.0

    standard_count = 0
    standard_tokens = 0
    standard_base_cost = 0.0
    standard_opt_cost = 0.0
    standard_energy_wh = 0.0

    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        is_reasoning = bool(int(num(r.get("is_reasoning", "0"))))
        
        tokens = inp + out
        total_tokens += tokens

        # BASELINE: naive deployment — everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        req_base_cost = pricing.request_cost(inp, out, lin, lout)
        base_cost += req_base_cost

        # OPTIMIZED: cascade (route_tier), prompt caching, batch API
        pin, pout = MODEL_PRICES[r["route_tier"]]
        req_opt_cost = pricing.request_cost(inp, out, pin, pout, cached_in=cached, batch=is_batch)
        opt_cost += req_opt_cost

        # Energy consumption
        req_energy_wh = sustainability.wh_per_query(tokens, is_reasoning=is_reasoning)

        if is_reasoning:
            reasoning_count += 1
            reasoning_tokens += tokens
            reasoning_base_cost += req_base_cost
            reasoning_opt_cost += req_opt_cost
            reasoning_energy_wh += req_energy_wh
        else:
            standard_count += 1
            standard_tokens += tokens
            standard_base_cost += req_base_cost
            standard_opt_cost += req_opt_cost
            standard_energy_wh += req_energy_wh

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")
        print("\n== Reasoning vs Standard Breakdown ==")
        print(f"Reasoning: count={reasoning_count} | tokens={reasoning_tokens:,} | base_cost=${reasoning_base_cost:.2f}/day | opt_cost=${reasoning_opt_cost:.2f}/day | energy={reasoning_energy_wh:,.1f} Wh")
        print(f"Standard : count={standard_count} | tokens={standard_tokens:,} | base_cost=${standard_base_cost:.2f}/day | opt_cost=${standard_opt_cost:.2f}/day | energy={standard_energy_wh:,.1f} Wh")

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "reasoning_stats": {
            "count": reasoning_count,
            "tokens": reasoning_tokens,
            "base_cost": round(reasoning_base_cost, 2),
            "opt_cost": round(reasoning_opt_cost, 2),
            "energy_wh": round(reasoning_energy_wh, 2),
        },
        "standard_stats": {
            "count": standard_count,
            "tokens": standard_tokens,
            "base_cost": round(standard_base_cost, 2),
            "opt_cost": round(standard_opt_cost, 2),
            "energy_wh": round(standard_energy_wh, 2),
        }
    }


if __name__ == "__main__":
    run()
