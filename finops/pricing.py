"""Pricing & purchasing economics — measure in $/1M-token, not $/GPU-hr.

Figures are June-2026 as-of snapshots from the deck's RESEARCH dossier; treat
live prices as fast-moving (re-baseline before each cohort).
"""
from __future__ import annotations


def request_cost(
    input_tok: int,
    output_tok: int,
    price_in_per_m: float,
    price_out_per_m: float,
    cached_in: int = 0,
    cache_discount: float = 0.10,   # Anthropic cached-read ~0.1x (=-90%)
    batch: bool = False,
    batch_discount: float = 0.50,   # Batch API ~ -50%
) -> float:
    """USD cost of a single request. Cached input billed at cache_discount x price."""
    cached_in = min(max(0, cached_in), input_tok)
    uncached_in = input_tok - cached_in
    cost = (
        (uncached_in / 1e6) * price_in_per_m
        + (cached_in / 1e6) * price_in_per_m * cache_discount
        + (output_tok / 1e6) * price_out_per_m
    )
    if batch:
        cost *= batch_discount
    return cost


def dollars_per_million(total_cost_usd: float, total_tokens: int) -> float:
    """Aggregate unit economics: $ per 1,000,000 tokens served."""
    if total_tokens <= 0:
        return 0.0
    return total_cost_usd / (total_tokens / 1e6)


def discount_stack(
    batch: bool = False,
    cache_hit_frac: float = 0.0,
    batch_discount: float = 0.50,
    cache_discount: float = 0.10,
) -> float:
    """Effective fraction of the naive bill after stacking discounts (input-heavy view).

    Discounts MULTIPLY: cache applies to the cached share of input, batch to the
    whole bill. batch + 100% cache-hit -> 0.5 * 0.1 = 0.05 (~95% off).
    """
    cache_mult = cache_hit_frac * cache_discount + (1.0 - cache_hit_frac)
    batch_mult = batch_discount if batch else 1.0
    return cache_mult * batch_mult


def break_even_utilization(discount_frac: float) -> float:
    """Utilization at which a commitment pays off ~= 1 - discount.

    A 45% reserved discount needs ~55% utilization (~13.2h/day) to beat on-demand.
    """
    return max(0.0, min(1.0, 1.0 - discount_frac))


def recommend_tier(
    hours_per_day: float,
    interruptible: bool,
    reserved_discount: float = 0.45,
    gpu_type: str | None = None,
    job_days: float | None = None,
    catalog: dict | None = None,
) -> str:
    """Pick a purchasing tier from a workload's duty cycle + interruptibility.

    Advanced policy:
      - Considers lock-in/obsolescence risk for cutting edge GPUs (like H100, H200, B200) vs utility GPUs.
      - Short-lived workloads: if job_days is short (e.g. < 30 days) and it's not a continuous inference kind,
        avoid reserved commitments and prefer on-demand or spot.
      - Interruption rate and rework overhead: if spot interruption rate is too high and rework makes it
        more expensive than reserved or on-demand, fallback.
    """
    # 1. Interruption rate by GPU type (higher for older/smaller GPUs like A10G/L4, lower for enterprise H100)
    INTERRUPT_RATES = {
        "H100": 0.02,
        "H200": 0.02,
        "B200": 0.03,
        "A100": 0.04,
        "MI300X": 0.04,
        "A10G": 0.07,
        "L4": 0.08,
    }

    # If no advanced metadata is provided, fall back to basic logic to preserve compatibility and test stability.
    if gpu_type is None or catalog is None or gpu_type not in catalog:
        duty = max(0.0, hours_per_day) / 24.0
        be = break_even_utilization(reserved_discount)
        if interruptible and hours_per_day < 24:
            return "spot"
        if duty >= be:
            return "reserved"
        return "on_demand"

    c = catalog[gpu_type]
    od_price = float(c["on_demand_hr"])
    spot_price = float(c["spot_hr"])
    r3_price = float(c["reserved_3yr_hr"])
    r1_price = float(c["reserved_1yr_hr"])

    days = job_days if job_days is not None else 30.0
    total_active_hours = hours_per_day * days

    # Calculate expected spot cost if interruptible
    spot_cost = float('inf')
    if interruptible:
        ir = INTERRUPT_RATES.get(gpu_type, 0.05)
        sim = spot_checkpoint_cost(
            job_hours=total_active_hours,
            spot_hr=spot_price,
            on_demand_hr=od_price,
            interrupt_rate=ir
        )
        spot_cost = sim["spot_cost"]

    # Lock-in risk and duration check for reserved
    # For reserved, you commit to paying 24/7 for the duration of the workload/job.
    is_cutting_edge = gpu_type in ["H100", "H200", "B200"]
    reserved_rate = r1_price if is_cutting_edge else r3_price
    reserved_cost = (24.0 * days) * reserved_rate
    on_demand_cost = total_active_hours * od_price

    # Compare costs to recommend the cheapest tier
    costs = {
        "on_demand": on_demand_cost,
        "reserved": reserved_cost,
    }
    if interruptible:
        costs["spot"] = spot_cost

    best_tier = min(costs, key=costs.get)
    return best_tier


def cache_is_worth_it(
    avg_cache_reads: float,
    write_cost_per_m: float,
    read_discount: float = 0.10,
) -> bool:
    """Evaluate if prompt caching saves money.

    Caching is worth it if:
    avg_cache_reads * (1 - read_discount) * input_cost > write_cost
    If we assume write_cost is equal to input_cost per million tokens, this simplifies to:
    avg_cache_reads * (1 - read_discount) > 1 => avg_cache_reads > 1 / (1 - read_discount).
    With read_discount = 0.10, threshold is ~1.11 reads.
    """
    if avg_cache_reads <= 0:
        return False
    return avg_cache_reads * (1.0 - read_discount) > 1.0



def spot_checkpoint_cost(
    job_hours: float,
    spot_hr: float,
    on_demand_hr: float,
    interrupt_rate: float = 0.05,      # per-hour chance (H100 spot ~<5%)
    ckpt_overhead_frac: float = 0.03,  # steady cost of writing checkpoints
    rework_hours_per_interrupt: float = 0.5,
) -> dict:
    """Effective cost of running a checkpointable job on spot vs on-demand.

    Interruptions waste the compute since the last checkpoint (rework); checkpointing
    adds a small steady overhead. Spot still wins for interruptible jobs.
    """
    expected_interrupts = job_hours * interrupt_rate
    rework_hours = expected_interrupts * rework_hours_per_interrupt
    effective_hours = job_hours * (1.0 + ckpt_overhead_frac) + rework_hours
    spot_cost = effective_hours * spot_hr
    on_demand_cost = job_hours * on_demand_hr
    savings_pct = (1.0 - spot_cost / on_demand_cost) * 100.0 if on_demand_cost > 0 else 0.0
    return {
        "spot_effective_hours": round(effective_hours, 2),
        "spot_cost": round(spot_cost, 2),
        "on_demand_cost": round(on_demand_cost, 2),
        "savings_pct": round(savings_pct, 1),
    }
