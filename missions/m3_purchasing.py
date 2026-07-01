"""M3 — Purchasing Strategy: break-even, tier choice, spot-checkpoint sim (deck §4).

Run: python missions/m3_purchasing.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing

DAYS = 30


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    on_demand_monthly = optimized_monthly = 0.0
    recs = []
    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        gpu_hours = hpd * DAYS * ngpu
        od = num(c["on_demand_hr"])
        on_demand_cost = gpu_hours * od

        j_days = int(num(j["days"]))
        tier = pricing.recommend_tier(hpd, interruptible, gpu_type=gtype, job_days=j_days, catalog=cat)
        if tier == "spot":
            sim = pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), od)
            opt_cost = sim["spot_cost"]
        elif tier == "reserved":
            # If cutting-edge GPU, commit to 1-year reserved to avoid lock-in, else 3-year reserved.
            is_cutting_edge = gtype in ["H100", "H200", "B200"]
            price_key = "reserved_1yr_hr" if is_cutting_edge else "reserved_3yr_hr"
            opt_cost = gpu_hours * num(c[price_key])
        else:
            opt_cost = on_demand_cost

        on_demand_monthly += on_demand_cost
        optimized_monthly += opt_cost
        recs.append({"job_id": j["job_id"], "gpu_type": gtype, "tier": tier,
                     "on_demand": round(on_demand_cost), "optimized": round(opt_cost)})

    savings = on_demand_monthly - optimized_monthly
    savings_pct = savings / on_demand_monthly * 100 if on_demand_monthly else 0.0

    from finops import sustainability
    carbon_savings = sustainability.calculate_carbon_savings(jobs, cat)

    if verbose:
        print("== M3 Purchasing Strategy ==")
        print(f"break-even utilization @ 45% reserved discount = {pricing.break_even_utilization(0.45):.0%}")
        print(f"{'job':18}{'gpu':7}{'tier':11}{'on-demand':>12}{'optimized':>12}")
        for r in recs:
            print(f"{r['job_id']:18}{r['gpu_type']:7}{r['tier']:11}${r['on_demand']:>11,}${r['optimized']:>11,}")
        print(f"\nmonthly: on-demand ${on_demand_monthly:,.0f} -> optimized ${optimized_monthly:,.0f}  ({savings_pct:.1f}% saved)")
        
        print("\n== Carbon-Aware Scheduling (Rescheduling Interruptible Jobs to europe-north1) ==")
        print(f"Rescheduled jobs: {[j['job_id'] for j in jobs if bool(int(num(j['interruptible'])))]}")
        print(f"Grid emissions: us-east-1 = {carbon_savings['us_carbon_g']/1e6:,.4f} tCO2e | europe-north1 = {carbon_savings['clean_carbon_g']/1e6:,.4f} tCO2e")
        print(f"Carbon savings: {carbon_savings['carbon_saved_g']/1e6:,.4f} tCO2e ({ (1 - carbon_savings['clean_carbon_g']/carbon_savings['us_carbon_g']) * 100:.1f}% reduction)")
        print(f"Electricity cost: us-east-1 = ${carbon_savings['us_cost_usd']:,.2f} | europe-north1 = ${carbon_savings['clean_cost_usd']:,.2f}")
        print(f"Electricity cost savings: ${carbon_savings['cost_saved_usd']:,.2f} ({ (1 - carbon_savings['clean_cost_usd']/carbon_savings['us_cost_usd']) * 100:.1f}% saved)")

    return {"recommendations": recs, "on_demand_monthly": round(on_demand_monthly),
            "optimized_monthly": round(optimized_monthly), "savings_pct": round(savings_pct, 1),
            "carbon_savings": carbon_savings}


if __name__ == "__main__":
    run()
