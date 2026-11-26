"""
Exact profit engine for the SkyCity dataset.

Every profit column in the dataset is built from the same simple formulas
(checked on all 1,696 rows, to within half a cent):

    revenue (per channel)  = orders (per channel) x AOV
    In-store profit        = revenue x (1 - COGS rate - OPEX rate)
    Uber Eats / DoorDash   = revenue x (1 - COGS rate - OPEX rate - commission rate)
    Self-delivery profit   = revenue x (1 - COGS rate - OPEX rate) - orders x delivery cost per order

Because profit is a known formula, we calculate it directly instead of
training a machine-learning model to guess it. Scenarios ("what if
commission rises 5 points?") are exact recalculations, not predictions.
"""
import numpy as np
import pandas as pd

CHANNELS = ["In-Store", "Uber Eats", "DoorDash", "Self-Delivery"]
ORDER_COLS = {
    "In-Store": "InStoreOrders",
    "Uber Eats": "UberEatsOrders",
    "DoorDash": "DoorDashOrders",
    "Self-Delivery": "SelfDeliveryOrders",
}
REVENUE_COLS = {
    "In-Store": "InStoreRevenue",
    "Uber Eats": "UberEatsRevenue",
    "DoorDash": "DoorDashRevenue",
    "Self-Delivery": "SelfDeliveryRevenue",
}
PROFIT_COLS = {
    "In-Store": "InStoreNetProfit",
    "Uber Eats": "UberEatsNetProfit",
    "DoorDash": "DoorDashNetProfit",
    "Self-Delivery": "SelfDeliveryNetProfit",
}
AGGREGATORS = ["Uber Eats", "DoorDash"]


def channel_profit(orders, aov, cogs, opex, commission, delivery_cost):
    """Profit for each channel from the formula. Works on scalars or arrays.

    orders: dict channel -> orders (scalar or array)
    Returns dict channel -> (revenue, profit).
    """
    base_margin = 1 - cogs - opex
    out = {}
    for ch in CHANNELS:
        rev = orders[ch] * aov
        if ch in AGGREGATORS:
            prof = rev * (base_margin - commission)
        elif ch == "Self-Delivery":
            prof = rev * base_margin - orders[ch] * delivery_cost
        else:
            prof = rev * base_margin
        out[ch] = (rev, prof)
    return out


def run_scenario(df, commission_pp=0.0, cogs_pp=0.0, opex_pp=0.0,
                 delivery_cost_pct=0.0, aov_pct=0.0, orders_pct=0.0,
                 shift_from=None, shift_to=None, shift_frac=0.0):
    """Recalculate every restaurant's profit under a scenario.

    *_pp  = change in percentage points (e.g. commission_pp=5 -> 30% becomes 35%)
    *_pct = percent change (e.g. aov_pct=10 -> AOV x 1.10)
    shift_* = move a fraction of one channel's orders to another channel.

    Returns a per-channel summary table (orders, revenue, profit, margin, profit/order).
    """
    orders = {ch: df[ORDER_COLS[ch]].astype(float) * (1 + orders_pct / 100) for ch in CHANNELS}
    if shift_from and shift_to and shift_from != shift_to and shift_frac > 0:
        moved = orders[shift_from] * shift_frac
        orders[shift_from] = orders[shift_from] - moved
        orders[shift_to] = orders[shift_to] + moved

    res = channel_profit(
        orders,
        aov=df["AOV"] * (1 + aov_pct / 100),
        cogs=df["COGSRate"] + cogs_pp / 100,
        opex=df["OPEXRate"] + opex_pp / 100,
        commission=df["CommissionRate"] + commission_pp / 100,
        delivery_cost=df["DeliveryCostPerOrder"] * (1 + delivery_cost_pct / 100),
    )
    rows = []
    for ch in CHANNELS:
        rev, prof = res[ch]
        o = orders[ch].sum()
        rows.append({
            "Channel": ch,
            "Orders": o,
            "Revenue": rev.sum(),
            "Net Profit": prof.sum(),
        })
    t = pd.DataFrame(rows)
    t["Margin (%)"] = np.where(t["Revenue"] > 0, t["Net Profit"] / t["Revenue"] * 100, 0.0)
    t["Profit / Order"] = np.where(t["Orders"] > 0, t["Net Profit"] / t["Orders"], 0.0)
    return t


def total_profit(df, **scenario):
    return run_scenario(df, **scenario)["Net Profit"].sum()


def formula_check(df):
    """Largest gap ($) between the dataset's profit columns and the formula."""
    orders = {ch: df[ORDER_COLS[ch]] for ch in CHANNELS}
    res = channel_profit(orders, df["AOV"], df["COGSRate"], df["OPEXRate"],
                         df["CommissionRate"], df["DeliveryCostPerOrder"])
    gap = 0.0
    for ch in CHANNELS:
        gap = max(gap, float((res[ch][1] - df[PROFIT_COLS[ch]]).abs().max()))
    return gap


def breakeven_commission_pp(df):
    """How many percentage points commission can rise before total profit hits $0.

    Profit falls by (aggregator revenue) for every +1.00 of commission, so
    break-even = current profit / aggregator revenue. Negative = already losing money.
    """
    profit = df["TotalNetProfit"].sum()
    agg_rev = df["UberEatsRevenue"].sum() + df["DoorDashRevenue"].sum()
    if agg_rev <= 0:
        return None
    return profit / agg_rev * 100


SENSITIVITY_DRIVERS = [
    # label, scenario kwargs for a +1 step, unit text
    ("OPEX rate", {"opex_pp": 1}, "+1 point"),
    ("COGS rate", {"cogs_pp": 1}, "+1 point"),
    ("Commission rate", {"commission_pp": 1}, "+1 point"),
    ("Average order value", {"aov_pct": 1}, "+1%"),
    ("Monthly orders", {"orders_pct": 1}, "+1%"),
    ("Delivery cost per order", {"delivery_cost_pct": 1}, "+1%"),
]


def sensitivity_table(df):
    """Exact $ change in monthly profit for a one-step change in each driver."""
    base = df["TotalNetProfit"].sum()
    rows = []
    for label, kw, unit in SENSITIVITY_DRIVERS:
        rows.append({"Driver": label, "Change": unit,
                     "Profit change ($)": total_profit(df, **kw) - base})
    t = pd.DataFrame(rows)
    t["Size"] = t["Profit change ($)"].abs()
    return t.sort_values("Size", ascending=False).drop(columns="Size").reset_index(drop=True)


def money_axis(v, _=None):
    """Axis label like $1.2M / -$500K (sign before the $)."""
    sign = "-" if v < 0 else ""
    v = abs(v)
    if v >= 1e6:
        return f"{sign}${v/1e6:,.1f}M"
    return f"{sign}${v/1e3:,.0f}K"


def signed_money(v):
    """+$1,234 / -$1,234; tiny rounding noise shows as $0."""
    if abs(v) < 0.5:
        return "$0"
    return f"{'+' if v > 0 else '-'}${abs(v):,.0f}"
