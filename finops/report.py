"""Report assembly — the lab's deliverable: baseline vs optimized + savings chart."""
from __future__ import annotations


def build_report(baseline_usd: float, optimized_usd: float, levers: dict,
                 sustainability: dict | None = None, period: str = "monthly",
                 reasoning_stats: dict | None = None, standard_stats: dict | None = None,
                 carbon_savings: dict | None = None) -> str:
    """Return a markdown cost-optimization report."""
    savings = baseline_usd - optimized_usd
    pct = (savings / baseline_usd * 100.0) if baseline_usd > 0 else 0.0
    lines = [
        "# NimbusAI — GPU Cost Optimization Report",
        "",
        f"**Period:** {period}  ",
        f"**Baseline spend:** ${baseline_usd:,.0f}  ",
        f"**Optimized spend:** ${optimized_usd:,.0f}  ",
        f"**Projected savings:** ${savings:,.0f}  (**{pct:.0f}%**)",
        "",
        "## Savings by lever",
        "",
        "| Lever | Savings (USD) |",
        "|---|---|",
    ]
    for name, amount in levers.items():
        lines.append(f"| {name} | ${amount:,.0f} |")
    if sustainability:
        lines += [
            "",
            "## Sustainability",
            "",
            f"- Energy per query: {sustainability.get('wh_per_query', 0):.2f} Wh",
            f"- Carbon per query: {sustainability.get('carbon_g', 0):.3f} gCO2e",
            f"- Cheapest+cleanest region: {sustainability.get('best_region', 'n/a')}",
        ]
    if reasoning_stats and standard_stats:
        lines += [
            "",
            "## Reasoning Budget & Energy Breakdown",
            "",
            "### Reasoning Queries",
            f"- **Count:** {reasoning_stats['count']} queries",
            f"- **Tokens:** {reasoning_stats['tokens']:,} tokens",
            f"- **Baseline Cost:** ${reasoning_stats['base_cost'] * 30:,.2f}/month",
            f"- **Optimized Cost:** ${reasoning_stats['opt_cost'] * 30:,.2f}/month",
            f"- **Total Energy:** {reasoning_stats['energy_wh'] * 30 / 1000:,.2f} kWh/month",
            "",
            "### Standard Queries",
            f"- **Count:** {standard_stats['count']} queries",
            f"- **Tokens:** {standard_stats['tokens']:,} tokens",
            f"- **Baseline Cost:** ${standard_stats['base_cost'] * 30:,.2f}/month",
            f"- **Optimized Cost:** ${standard_stats['opt_cost'] * 30:,.2f}/month",
            f"- **Total Energy:** {standard_stats['energy_wh'] * 30 / 1000:,.2f} kWh/month",
            "",
            "> [!TIP]",
            "> **FinOps Routing Recommendation:** Reasoning models consume approximately 80x more energy per query than standard models. We recommend routing simple queries (e.g. standard classification, summarization) to small models, reserving reasoning models only for high-complexity prompts where logical confidence is low.",
        ]
    if carbon_savings:
        us_c = carbon_savings['us_carbon_g']
        clean_c = carbon_savings['clean_carbon_g']
        reduction = (1 - clean_c / us_c) * 100 if us_c else 0.0
        us_cost = carbon_savings['us_cost_usd']
        clean_cost = carbon_savings['clean_cost_usd']
        cost_saved = carbon_savings['cost_saved_usd']
        cost_reduction = (1 - clean_cost / us_cost) * 100 if us_cost else 0.0

        lines += [
            "",
            "## Carbon-Aware Scheduling (Rescheduling Interruptible Workloads)",
            "",
            "Rescheduling flexible, interruptible training and batch jobs to **europe-north1** (Norway hydro) yields significant sustainability and financial returns:",
            "",
            f"- **Baseline emissions (us-east-1):** {us_c / 1e6:,.4f} tCO2e",
            f"- **Optimized emissions (europe-north1):** {clean_c / 1e6:,.4f} tCO2e",
            f"- **Carbon reduction:** {carbon_savings['carbon_saved_g'] / 1e6:,.4f} tCO2e (**{reduction:.1f}%**)",
            f"- **Baseline electricity cost (us-east-1):** ${us_cost:,.2f}",
            f"- **Optimized electricity cost (europe-north1):** ${clean_cost:,.2f}",
            f"- **Electricity cost savings:** ${cost_saved:,.2f} (**{cost_reduction:.1f}%**)",
        ]
    lines += ["", "_Figures are June-2026 as-of snapshots; re-baseline before acting._"]
    return "\n".join(lines)


def savings_waterfall(levers: dict, path: str) -> str:
    """Write a simple savings bar chart PNG. Returns the path. No-op if matplotlib absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    names = list(levers.keys())
    vals = [levers[n] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(names, vals, color="#2e548a")
    ax.set_ylabel("Savings (USD / month)")
    ax.set_title("GPU cost savings by FinOps lever")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
