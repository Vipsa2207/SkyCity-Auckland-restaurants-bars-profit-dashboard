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

st.set_page_config(page_title="Channel Shift | SkyCity", page_icon="🔀", layout="wide")
st.markdown(DARK_CSS, unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor": "#161927", "axes.facecolor": "#161927",
    "axes.edgecolor": "#2a2d3e",   "axes.labelcolor": "#e8eaf6",
    "xtick.color": "#9e9e9e",      "ytick.color": "#9e9e9e",
    "text.color": "#e8eaf6",       "grid.color": "#2a2d3e",
    "grid.linestyle": "--",        "grid.alpha": 0.4,
})
CH_COLORS = {"In-Store": "#06d6a0", "Uber Eats": "#ef476f", "DoorDash": "#f77f00", "Self-Delivery": "#7c83fd"}


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
page_title("Channel Shift Calculator",
           "What if some delivery-app customers ordered in-store or through the restaurant's own delivery instead?")

if fdf.empty:
    st.warning("No restaurants match the current filters.")
    st.stop()

base = fdf["TotalNetProfit"].sum()

section_header("Move Orders Between Channels")
c1, c2, c3 = st.columns(3)
with c1:
    src = st.selectbox("Move orders FROM", pe.AGGREGATORS, index=0)
with c2:
    dst = st.selectbox("Move them TO", ["Self-Delivery", "In-Store"], index=0)
with c3:
    frac = st.slider(f"Share of {src} orders moved (%)", 0, 50, 10, 1)

new = pe.total_profit(fdf, shift_from=src, shift_to=dst, shift_frac=frac / 100)
moved = fdf[pe.ORDER_COLS[src]].sum() * frac / 100
delta = new - base
pct = delta / abs(base) * 100 if base else 0.0

k1, k2, k3, k4 = st.columns(4)
with k1: kpi_card("Orders moved / month", f"{moved:,.0f}", f"{src} → {dst}", "info")
with k2: kpi_card("Current Profit", f"${base:,.0f}", "Per month", "info")
with k3: kpi_card("Profit After Shift", f"${new:,.0f}", "Exact recalculation", "positive" if new >= 0 else "negative")
with k4: kpi_card("Extra Profit", pe.signed_money(delta),
                  f"{pct:+.1f}% · ${delta / moved if moved else 0:,.2f} per moved order",
                  "info" if abs(delta) < 0.5 else ("positive" if delta > 0 else "negative"))

# ── Why each moved order is worth more ───────────────────────────────────────
section_header("Why Each Moved Order Earns More")
t = pe.run_scenario(fdf)
ppo = dict(zip(t["Channel"], t["Profit / Order"]))
fig, ax = plt.subplots(figsize=(9, 3.0))
chs = list(ppo.keys())
bars = ax.barh(chs, [ppo[c] for c in chs], color=[CH_COLORS[c] for c in chs], height=0.55)
ax.bar_label(bars, labels=[f"${ppo[c]:.2f}" for c in chs], padding=4, color="#e8eaf6", fontsize=9)
ax.invert_yaxis(); ax.set_xlabel("Average profit per order")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:.0f}"))
ax.set_xlim(min(0, min(ppo.values()) * 1.2), max(ppo.values()) * 1.18); ax.grid(axis="x")
plt.tight_layout(); st.pyplot(fig); plt.close()

# ── Curve: extra profit vs % moved ───────────────────────────────────────────
section_header("Extra Profit as More Orders Move")
steps = np.arange(0, 51, 5)
fig, ax = plt.subplots(figsize=(10, 3.4))
for s, d, style in [("Uber Eats", "Self-Delivery", "-"), ("Uber Eats", "In-Store", "--"),
                    ("DoorDash", "Self-Delivery", "-"), ("DoorDash", "In-Store", "--")]:
    gains = [pe.total_profit(fdf, shift_from=s, shift_to=d, shift_frac=p / 100) - base for p in steps]
    ax.plot(steps, gains, style, color=CH_COLORS[d] if s == "Uber Eats" else "#ffd166",
            linewidth=2, label=f"{s} → {d}")
ax.set_xlabel(f"Share of the platform's orders moved (%)"); ax.set_ylabel("Extra profit / month")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(pe.money_axis))
ax.legend(fontsize=8, ncol=2); ax.grid()
plt.tight_layout(); st.pyplot(fig); plt.close()

a1, a2 = st.columns(2)
with a1:
    insight_card("What this shows",
        "Each order moved off a delivery app keeps the commission (about "
        f"<b>{fdf['CommissionRate'].mean()*100:.0f}%</b> of the order). Moving it to self-delivery adds the "
        f"restaurant's own delivery cost (about <b>${fdf['DeliveryCostPerOrder'].mean():.2f}</b> per order).",
        "positive")
with a2:
    insight_card("What this does NOT show",
        "Whether customers would actually switch. Some app customers would simply stop ordering. "
        "Treat these numbers as the <b>upper bound</b> if every moved customer stays, and test with a small pilot "
        "(e.g. a discount for ordering direct) before planning on them.", "risk")
st.caption("Assumptions: moved orders keep the same order value; food and operating cost rates stay the same; "
           "self-delivery cost per order stays the same at higher volume; no extra staff or marketing cost is included.")
