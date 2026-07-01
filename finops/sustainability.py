"""Sustainability economics — energy and carbon as governed cost levers (deck §11).

Region selection cuts $ and carbon together; reasoning queries are an energy bomb.
"""
from __future__ import annotations

# Grid carbon intensity (gCO2 / kWh) — illustrative 2026 snapshot.
REGION_CARBON = {
    "us-east-1": 380,
    "us-west-2": 120,   # Oregon hydro
    "europe-north1": 30,  # Norway
    "europe-central2": 660,  # Poland (dirtiest)
    "us-east-wa": 90,
}
# Electricity price (USD / kWh) — illustrative.
REGION_PRICE_KWH = {
    "us-east-1": 0.12,
    "us-west-2": 0.07,
    "europe-north1": 0.09,
    "europe-central2": 0.18,
    "us-east-wa": 0.055,
}

REASONING_ENERGY_MULTIPLIER = 80.0  # deck: reasoning ~74-86x a small-model query


def wh_per_query(total_tokens: int, wh_per_1k_tokens: float = 0.30, is_reasoning: bool = False) -> float:
    """Energy for one query. Median Gemini prompt ~0.24 Wh; reasoning ~74-86x."""
    base = (total_tokens / 1000.0) * wh_per_1k_tokens
    return base * (REASONING_ENERGY_MULTIPLIER if is_reasoning else 1.0)


def carbon_g(wh: float, region: str = "us-east-1") -> float:
    """Grams CO2 for an energy amount in a region."""
    gco2_kwh = REGION_CARBON.get(region, 400)
    return (wh / 1000.0) * gco2_kwh


def energy_cost_usd(wh: float, region: str = "us-east-1") -> float:
    """Electricity cost of an energy amount in a region."""
    return (wh / 1000.0) * REGION_PRICE_KWH.get(region, 0.12)


def tokens_per_watt(total_tokens: int, wh: float, seconds: float = 1.0) -> float:
    """Energy efficiency of serving: tokens per watt (higher is better)."""
    watts = (wh * 3600.0) / seconds if seconds > 0 else 0.0
    return total_tokens / watts if watts > 0 else 0.0


def calculate_carbon_savings(workloads: list[dict], catalog: dict) -> dict:
    """Calculate potential carbon & electricity savings from migrating interruptible jobs to europe-north1.

    Returns a dict with us_carbon_g, clean_carbon_g, carbon_saved_g, us_cost_usd, clean_cost_usd, cost_saved_usd.
    """
    us_carbon = 0.0
    clean_carbon = 0.0
    us_cost = 0.0
    clean_cost = 0.0

    for j in workloads:
        # Check if the workload is interruptible (can be rescheduled)
        try:
            val = j.get("interruptible", "0")
            interruptible = bool(int(float(val)))
        except ValueError:
            interruptible = False

        if not interruptible:
            continue

        gtype = j["gpu_type"]
        if gtype not in catalog:
            continue

        try:
            ngpu = int(float(j["num_gpus"]))
            hpd = float(j["hours_per_day"])
            days = float(j["days"])
        except ValueError:
            continue

        if days <= 0:
            days = 30.0

        # active GPU hours
        active_hours = hpd * days * ngpu
        watts = float(catalog[gtype].get("watts", 0.0))

        # Total energy in Wh
        energy_wh = active_hours * watts

        # Carbon in grams
        us_carbon += carbon_g(energy_wh, "us-east-1")
        clean_carbon += carbon_g(energy_wh, "europe-north1")

        # Electricity cost in USD
        us_cost += energy_cost_usd(energy_wh, "us-east-1")
        clean_cost += energy_cost_usd(energy_wh, "europe-north1")

    return {
        "us_carbon_g": round(us_carbon, 2),
        "clean_carbon_g": round(clean_carbon, 2),
        "carbon_saved_g": round(us_carbon - clean_carbon, 2),
        "us_cost_usd": round(us_cost, 2),
        "clean_cost_usd": round(clean_cost, 2),
        "cost_saved_usd": round(us_cost - clean_cost, 2),
    }

