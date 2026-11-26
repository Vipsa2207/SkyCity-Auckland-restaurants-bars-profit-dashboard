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

st.set_page_config(page_title="Sensitivity | SkyCity", page_icon="📈", layout="wide")
st.markdown(DARK_CSS, unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor": "#161927", "axes.facecolor": "#161927",
    "axes.edgecolor": "#2a2d3e",   "axes.labelcolor": "#e8eaf6",
    "xtick.color": "#9e9e9e",      "ytick.color": "#9e9e9e",
    "text.color": "#e8eaf6",       "grid.color": "#2a2d3e",
    "grid.linestyle": "--",        "grid.alpha": 0.4,
})


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
page_title("Sensitivity",
           "Which lever moves profit the most? Each bar is an exact recalculation for one small change.")

if fdf.empty:
    st.warning("No restaurants match the current filters.")
    st.stop()

base = fdf["TotalNetProfit"].sum()
sens = pe.sensitivity_table(fdf)

section_header("Profit Change from One Small Step")
fig, ax = plt.subplots(figsize=(10, 3.8))
labels = [f"{d} ({c})" for d, c in zip(sens["Driver"], sens["Change"])]
vals = sens["Profit change ($)"]
bars = ax.barh(labels, vals, color=["#06d6a0" if v >= 0 else "#ef476f" for v in vals], height=0.55)
ax.bar_label(bars, labels=[f"{'+' if v >= 0 else '−'}${abs(v):,.0f}" for v in vals], padding=4,
             color="#e8eaf6", fontsize=9)
ax.axvline(0, color="#9e9e9e", linewidth=0.8)
ax.invert_yaxis(); ax.set_xlabel("Change in monthly profit")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(pe.money_axis))
lim = vals.abs().max() * 1.35
ax.set_xlim(-lim, lim); ax.grid(axis="x")
plt.tight_layout(); st.pyplot(fig); plt.close()
st.caption("Points = percentage points of revenue (e.g. OPEX 38% → 39%). Percent = relative change (e.g. AOV \\$38 → \\$38.38). "
           "Food cost (COGS) and operating cost (OPEX) have identical effects because both are charged on all revenue.")

st.dataframe(sens.style.format({"Profit change ($)": "{:+,.0f}"}), width="stretch", hide_index=True)

# ── Insights ─────────────────────────────────────────────────────────────────
section_header("What This Means")
opex = sens.loc[sens["Driver"] == "OPEX rate", "Profit change ($)"].iloc[0]
comm = sens.loc[sens["Driver"] == "Commission rate", "Profit change ($)"].iloc[0]
cost_ratio = (fdf["COGSRate"] + fdf["OPEXRate"]).mean() * 100
i1, i2 = st.columns(2)
with i1:
    if comm != 0:
        insight_card("Costs beat commission",
            f"One point of operating cost costs <b>${abs(opex):,.0f}</b> a month; one point of commission costs "
            f"<b>${abs(comm):,.0f}</b>. That's <b>{abs(opex / comm):.1f}×</b> more, because operating cost applies to every "
            "order while commission applies only to delivery-app orders.", "positive")
    else:
        insight_card("No delivery-app revenue", "Commission has no effect on this selection.", "positive")
with i2:
    avg_comm = fdf["CommissionRate"].mean() * 100
    left = 100 - cost_ratio - avg_comm
    insight_card("Cost structure",
        f"Food + operating costs average <b>{cost_ratio:.1f}%</b> of revenue in this selection. Add a "
        f"<b>{avg_comm:.0f}%</b> commission and a delivery-app order keeps only about <b>{left:.1f}%</b> of its revenue"
        + (" &mdash; it loses money." if left < 0 else "."), "risk" if left < 5 else "positive")

seg = (fdf.groupby("Segment")
       .apply(lambda g: pd.Series({
           "Restaurants": len(g),
           "Loss-making": int((g["TotalNetProfit"] < 0).sum()),
           "Food + Op. cost (%)": (g["COGSRate"] + g["OPEXRate"]).mean() * 100,
           "Margin (%)": g["TotalNetProfit"].sum() / g["TotalRevenue"].sum() * 100,
       }), include_groups=False)
       .reset_index().sort_values("Margin (%)"))
section_header("Margin by Segment")
st.dataframe(seg.style.format({"Restaurants": "{:,.0f}", "Loss-making": "{:,.0f}",
                               "Food + Op. cost (%)": "{:.1f}%", "Margin (%)": "{:.1f}%"}),
             width="stretch", hide_index=True)
