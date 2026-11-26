import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from app.utils.data_loader import load_data, get_cuisines, get_segments, get_subregions
from app.utils.plot_utils import (render_header, render_sidebar_filters, DARK_CSS,
                                   kpi_card, section_header, page_title, insight_card)
from app.utils import profit_engine as pe

st.set_page_config(page_title="What-If Simulator | SkyCity", page_icon="🎛️", layout="wide")
st.markdown(DARK_CSS, unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor": "#161927", "axes.facecolor": "#161927",
    "axes.edgecolor": "#2a2d3e",   "axes.labelcolor": "#e8eaf6",
    "xtick.color": "#9e9e9e",      "ytick.color": "#9e9e9e",
    "text.color": "#e8eaf6",       "grid.color": "#2a2d3e",
    "grid.linestyle": "--",        "grid.alpha": 0.4,
})
PALETTE = ["#7c83fd", "#f77f00", "#06d6a0", "#ef476f", "#ffd166"]


@st.cache_data
def get_data():
    return load_data()


df = get_data()
cuisines, segments, subregions = render_sidebar_filters(df, get_cuisines, get_segments, get_subregions)
fdf = df.copy()
if cuisines:   fdf = fdf[fdf["CuisineType"].isin(cuisines)]
if segments:   fdf = fdf[fdf["Segment"].isin(segments)]
if subregions: fdf = fdf[fdf["Subregion"].isin(subregions)]

render_header()
page_title("What-If Simulator",
           "Change costs, prices or volume for every restaurant in the selection and recalculate profit exactly.")

if fdf.empty:
    st.warning("No restaurants match the current filters.")
    st.stop()

base = fdf["TotalNetProfit"].sum()

# ── Scenario inputs ──────────────────────────────────────────────────────────
section_header("Scenario Settings")
st.caption("Point changes are added to each restaurant's own rate (e.g. +5 points turns a 30% commission into 35%). "
           "Percent changes multiply (e.g. +10% AOV turns \\$38 into \\$41.80).")
s1, s2, s3 = st.columns(3)
with s1:
    st.markdown("**Delivery platforms**")
    comm_pp = st.slider("Commission change (points)", -15, 15, 0, 1)
    dc_pct = st.slider("Delivery cost per order (% change)", -50, 50, 0, 5)
with s2:
    st.markdown("**Costs**")
    cogs_pp = st.slider("Food cost rate / COGS (points)", -10, 10, 0, 1)
    opex_pp = st.slider("Operating cost rate / OPEX (points)", -10, 10, 0, 1)
with s3:
    st.markdown("**Demand**")
    aov_pct = st.slider("Average order value (% change)", -20, 20, 0, 1)
    ord_pct = st.slider("Monthly orders (% change)", -30, 30, 0, 1)

scn = dict(commission_pp=comm_pp, delivery_cost_pct=dc_pct, cogs_pp=cogs_pp,
           opex_pp=opex_pp, aov_pct=aov_pct, orders_pct=ord_pct)
new_t = pe.run_scenario(fdf, **scn)
new = new_t["Net Profit"].sum()
delta = new - base
pct = delta / abs(base) * 100 if base else 0.0

section_header("Result vs Current")
k1, k2, k3 = st.columns(3)
with k1: kpi_card("Current Monthly Profit", f"${base:,.0f}", "From the dataset", "info")
with k2: kpi_card("Scenario Monthly Profit", f"${new:,.0f}", "Exact recalculation",
                  "positive" if new >= 0 else "negative")
with k3: kpi_card("Change", pe.signed_money(delta),
                  f"{0.0 if abs(delta) < 0.5 else pct:+.1f}% vs current", "info" if abs(delta) < 0.5 else ("positive" if delta > 0 else "negative"))

cur_t = pe.run_scenario(fdf)
comp = pd.DataFrame({"Channel": cur_t["Channel"], "Current": cur_t["Net Profit"], "Scenario": new_t["Net Profit"]})
comp["Change"] = (comp["Scenario"] - comp["Current"]).where(lambda x: x.abs() >= 0.5, 0.0)
c1, c2 = st.columns([1, 1.2])
with c1:
    st.dataframe(comp.style.format({"Current": "${:,.0f}", "Scenario": "${:,.0f}", "Change": "{:+,.0f}"}),
                 width="stretch", hide_index=True)
with c2:
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    x = np.arange(len(comp))
    ax.bar(x - 0.2, comp["Current"], 0.38, color="#5b6075", label="Current")
    ax.bar(x + 0.2, comp["Scenario"], 0.38, color=PALETTE[0], label="Scenario")
    ax.set_xticks(x); ax.set_xticklabels(comp["Channel"])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(pe.money_axis))
    ax.axhline(0, color="#9e9e9e", linewidth=0.8)
    ax.set_title("Profit by channel", fontsize=11); ax.legend(fontsize=8); ax.grid(axis="y")
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ── Commission sweep ─────────────────────────────────────────────────────────
section_header("How Much Commission Can the Selection Absorb?")
be_pp = pe.breakeven_commission_pp(fdf)
cur_comm = fdf["CommissionRate"].mean() * 100
sweep = np.arange(-15, 31, 1)
profits = [pe.total_profit(fdf, commission_pp=float(p)) for p in sweep]
fig, ax = plt.subplots(figsize=(10, 3.6))
ax.plot(cur_comm + sweep, profits, color=PALETTE[0], linewidth=2.2)
ax.axhline(0, color=PALETTE[3], linestyle="--", linewidth=1.2, label="Break-even ($0 profit)")
ax.axvline(cur_comm, color=PALETTE[4], linestyle=":", linewidth=1.5, label=f"Current avg {cur_comm:.1f}%")
ax.set_xlabel("Average commission rate (%)"); ax.set_ylabel("Monthly profit")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(pe.money_axis))
ax.legend(fontsize=8); ax.grid()
plt.tight_layout(); st.pyplot(fig); plt.close()

if be_pp is None:
    st.info("This selection has no Uber Eats or DoorDash revenue, so commission does not affect profit.")
elif be_pp < 0:
    insight_card("Already losing money",
        f"This selection loses money at today's commission. Even a commission cut of {abs(be_pp):.1f} points "
        f"(to about {cur_comm + be_pp:.1f}%) would only bring it back to break-even. The problem is costs, not just commission.",
        "risk")
else:
    insight_card("Commission headroom",
        f"Profit reaches $0 if commission rises about <b>{be_pp:.1f} points</b> (to roughly "
        f"<b>{cur_comm + be_pp:.1f}%</b>). Every +1 point of commission costs "
        f"<b>${(fdf['UberEatsRevenue'].sum() + fdf['DoorDashRevenue'].sum()) / 100:,.0f}</b> a month.", "positive")
st.caption("This is a straight line because commission is charged as a fixed % of Uber Eats and DoorDash revenue. "
           "It assumes order volumes do not change when commission changes.")
