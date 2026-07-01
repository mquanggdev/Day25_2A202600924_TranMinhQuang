import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finops import pricing, sustainability
from missions import m2_inference_levers, m3_purchasing


def test_cache_is_worth_it():
    # 2.0 reads with 0.10 discount => 2 * 0.9 = 1.8 > 1.0 => True
    assert pricing.cache_is_worth_it(avg_cache_reads=2.0, write_cost_per_m=3.0) is True
    # 0.5 reads => False
    assert pricing.cache_is_worth_it(avg_cache_reads=0.5, write_cost_per_m=3.0) is False
    # Negative/Zero reads => False
    assert pricing.cache_is_worth_it(avg_cache_reads=0, write_cost_per_m=3.0) is False
    assert pricing.cache_is_worth_it(avg_cache_reads=-1, write_cost_per_m=3.0) is False


def test_recommend_tier_advanced():
    # Setup dummy catalog
    catalog = {
        "H100": {
            "on_demand_hr": 2.5,
            "spot_hr": 1.5,
            "reserved_1yr_hr": 2.0,
            "reserved_3yr_hr": 1.4,
            "watts": 700,
        },
        "A10G": {
            "on_demand_hr": 1.0,
            "spot_hr": 0.4,
            "reserved_1yr_hr": 0.8,
            "reserved_3yr_hr": 0.6,
            "watts": 150,
        }
    }
    
    # 1. Spiky work on H100 (high od vs reserved commit)
    # If job is 2 hours per day, 10 days, not interruptible
    tier = pricing.recommend_tier(hours_per_day=2.0, interruptible=False, gpu_type="H100", job_days=10, catalog=catalog)
    assert tier == "on_demand"

    # 2. High duty cycle on A10G (24h/day, 30 days) => reserved
    tier = pricing.recommend_tier(hours_per_day=24.0, interruptible=False, gpu_type="A10G", job_days=30, catalog=catalog)
    assert tier == "reserved"

    # 3. Interruptible work on H100 => spot (should select spot since spot price 1.5 is cheaper than od 2.5 and reserved_1yr 2.0)
    tier = pricing.recommend_tier(hours_per_day=8.0, interruptible=True, gpu_type="H100", job_days=14, catalog=catalog)
    assert tier == "spot"


def test_calculate_carbon_savings():
    catalog = {
        "H100": {"watts": 700},
        "A100": {"watts": 400},
    }
    workloads = [
        {"job_id": "job1", "gpu_type": "H100", "num_gpus": "2", "hours_per_day": "10", "days": "10", "interruptible": "1"},
        {"job_id": "job2", "gpu_type": "A100", "num_gpus": "1", "hours_per_day": "24", "days": "30", "interruptible": "0"}, # not interruptible
    ]
    res = sustainability.calculate_carbon_savings(workloads, catalog)
    # Active hours for job1: 10 * 10 * 2 = 200 hours.
    # Energy: 200 * 700 = 140,000 Wh = 140 kWh.
    # US Carbon (380 g/kWh): 140 * 380 = 53,200 g
    # Clean Carbon (30 g/kWh): 140 * 30 = 4,200 g
    # Carbon Saved: 49,000 g
    assert abs(res["us_carbon_g"] - 53200) < 1e-2
    assert abs(res["clean_carbon_g"] - 4200) < 1e-2
    assert abs(res["carbon_saved_g"] - 49000) < 1e-2

    # Electricity costs:
    # US Cost ($0.12/kWh): 140 * 0.12 = 16.80 USD
    # Clean Cost ($0.09/kWh): 140 * 0.09 = 12.60 USD
    # Cost Saved: 4.20 USD
    assert abs(res["us_cost_usd"] - 16.80) < 1e-2
    assert abs(res["clean_cost_usd"] - 12.60) < 1e-2
    assert abs(res["cost_saved_usd"] - 4.20) < 1e-2


def test_missions_return_new_stats():
    r2 = m2_inference_levers.run(verbose=False)
    assert "reasoning_stats" in r2
    assert "standard_stats" in r2
    assert r2["reasoning_stats"]["count"] > 0
    assert r2["standard_stats"]["count"] > 0

    r3 = m3_purchasing.run(verbose=False)
    assert "carbon_savings" in r3
    assert r3["carbon_savings"]["carbon_saved_g"] > 0
