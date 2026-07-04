import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import sys, os

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from app.utils.data_loader import load_data, get_cuisines, get_segments, get_subregions
from app.utils.plot_utils import (render_header, render_sidebar_filters, DARK_CSS,
                                   kpi_card, section_header, page_title, insight_card)

st.set_page_config(page_title="What-If Simulator | SkyCity", page_icon="🎮", layout="wide")
st.markdown(DARK_CSS, unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor": "#0f1117", "axes.facecolor": "#161927",
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
cuisines, segments, subregions = render_sidebar_filters(
    df, get_cuisines, get_segments, get_subregions)

fdf = df.copy()
if cuisines:   fdf = fdf[fdf["CuisineType"].isin(cuisines)]
if segments:   fdf = fdf[fdf["Segment"].isin(segments)]
if subregions: fdf = fdf[fdf["Subregion"].isin(subregions)]

render_header()
page_title("What-If Simulator")

# ── Train model silently ──────────────────────────────────────────────────────
@st.cache_data
def train_gb(data):
    d = data.copy()
    le = LabelEncoder()
    for col in ["CuisineType","Segment","Subregion"]:
        d[col+"_enc"] = le.fit_transform(d[col].astype(str))
    features = [
        "InStoreShare","UE_share","DD_share","SD_share",
        "CommissionRate","DeliveryCostPerOrder","DeliveryRadiusKM",
        "GrowthFactor","AOV","MonthlyOrders","COGSRate","OPEXRate",
        "Commission_UE","Commission_DD","DeliveryCost_SD",
        "CuisineType_enc","Segment_enc","Subregion_enc"
    ]
    target = "TotalNetProfit"
    d = d.dropna(subset=features+[target])
    X, y = d[features].values, d[target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    return model, d[features].mean().to_dict()

model, base_means = train_gb(fdf)

def predict(params):
    vec = np.array([[
        params["instore"], params["ue"], params["dd"], params["sd"],
        params["comm"], params["delcost"], params["radius"],
        params["growth"], params["aov"], params["orders"],
        params["cogs"], params["opex"],
        params["comm"]*params["ue"],
        params["comm"]*params["dd"],
        params["delcost"]*params["sd"],
        params["cuisine_enc"], params["segment_enc"], params["subregion_enc"]
    ]])
    return model.predict(vec)[0]

# ── Scenario Builder ──────────────────────────────────────────────────────────
section_header("Scenario Configuration")

tab1, tab2, tab3 = st.tabs([
    "📦 Channel Mix Shift",
    "💰 Commission & Cost Change",
    "📈 Growth & Demand Shift"
])

# Baseline values
base_params = {
    "instore": float(fdf["InStoreShare"].mean()),
    "ue":      float(fdf["UE_share"].mean()),
    "dd":      float(fdf["DD_share"].mean()),
    "sd":      float(fdf["SD_share"].mean()),
    "comm":    float(fdf["CommissionRate"].mean()),
    "delcost": float(fdf["DeliveryCostPerOrder"].mean()),
    "radius":  float(fdf["DeliveryRadiusKM"].mean()),
    "growth":  float(fdf["GrowthFactor"].mean()),
    "aov":     float(fdf["AOV"].mean()),
    "orders":  float(fdf["MonthlyOrders"].mean()),
    "cogs":    float(fdf["COGSRate"].mean()),
    "opex":    float(fdf["OPEXRate"].mean()),
    "cuisine_enc": 0, "segment_enc": 0, "subregion_enc": 0
}
baseline_profit = predict(base_params)

with tab1:
    st.markdown("**Adjust delivery channel shares to see profit impact**")
    c1, c2, c3, c4 = st.columns(4)
    with c1: t1_instore = st.slider("In-Store Share",      0.0, 1.0, base_params["instore"], 0.01, key="t1_is")
    with c2: t1_ue      = st.slider("Uber Eats Share",     0.0, 1.0, base_params["ue"],      0.01, key="t1_ue")
    with c3: t1_dd      = st.slider("DoorDash Share",      0.0, 1.0, base_params["dd"],      0.01, key="t1_dd")
    with c4: t1_sd      = st.slider("Self-Delivery Share", 0.0, 1.0, base_params["sd"],      0.01, key="t1_sd")
    scenario1 = {**base_params, "instore": t1_instore, "ue": t1_ue, "dd": t1_dd, "sd": t1_sd}
    active_scenario = scenario1
    active_label = "Channel Mix Shift"

with tab2:
    st.markdown("**Adjust commission rates and delivery costs**")
    c1, c2, c3 = st.columns(3)
    with c1: t2_comm    = st.slider("Commission Rate",         0.05, 0.45, base_params["comm"],    0.01, key="t2_comm")
    with c2: t2_delcost = st.slider("Delivery Cost/Order ($)", 0.5,  6.0,  base_params["delcost"], 0.10, key="t2_dc")
    with c3: t2_radius  = st.slider("Delivery Radius (km)",    2,    20,   int(base_params["radius"]), 1, key="t2_rad")
    scenario2 = {**base_params, "comm": t2_comm, "delcost": t2_delcost, "radius": t2_radius}
    active_scenario = scenario2
    active_label = "Commission & Cost Change"

with tab3:
    st.markdown("**Simulate demand growth or contraction**")
    c1, c2, c3 = st.columns(3)
    with c1: t3_growth = st.slider("Growth Factor",           0.90, 1.15, base_params["growth"], 0.01, key="t3_g")
    with c2: t3_orders = st.slider("Monthly Orders",          200,  3000, int(base_params["orders"]), 50, key="t3_o")
    with c3: t3_aov    = st.slider("Avg Order Value ($)",     25.0, 55.0, base_params["aov"],    0.50, key="t3_aov")
    scenario3 = {**base_params, "growth": t3_growth, "orders": t3_orders, "aov": t3_aov}
    active_scenario = scenario3
    active_label = "Growth & Demand Shift"

# ── Always show results for the last active tab ───────────────────────────────
# Run all three so user sees results regardless of tab
s1_profit = predict(scenario1)
s2_profit = predict(scenario2)
s3_profit = predict(scenario3)

st.markdown("---")

# ── Scenario Results ──────────────────────────────────────────────────────────
section_header("Scenario Results vs Baseline")
st.markdown("<p style='color:#9aa0b4; font-size:0.85rem; margin-bottom:1rem;'>Baseline reflects current filtered data average. Scenarios show predicted profit under each hypothetical configuration.</p>", unsafe_allow_html=True)

r1, r2, r3, r4 = st.columns(4)
with r1: kpi_card("Baseline Profit",       f"${baseline_profit:,.0f}", "Current average", "info")
with r2:
    delta1 = s1_profit - baseline_profit
    kpi_card("Channel Mix Shift",  f"${s1_profit:,.0f}",
             f"{'↑' if delta1>=0 else '↓'} ${abs(delta1):,.0f} vs baseline",
             "positive" if delta1>=0 else "negative")
with r3:
    delta2 = s2_profit - baseline_profit
    kpi_card("Commission & Cost",  f"${s2_profit:,.0f}",
             f"{'↑' if delta2>=0 else '↓'} ${abs(delta2):,.0f} vs baseline",
             "positive" if delta2>=0 else "negative")
with r4:
    delta3 = s3_profit - baseline_profit
    kpi_card("Growth & Demand",    f"${s3_profit:,.0f}",
             f"{'↑' if delta3>=0 else '↓'} ${abs(delta3):,.0f} vs baseline",
             "positive" if delta3>=0 else "negative")

st.markdown("<div style='margin-top:1.2rem;'>", unsafe_allow_html=True)

# ── Bar Chart Comparison ──────────────────────────────────────────────────────
section_header("Scenario Profit Comparison")

labels   = ["Baseline", "Channel Mix", "Commission & Cost", "Growth & Demand"]
values   = [baseline_profit, s1_profit, s2_profit, s3_profit]
colors   = ["#7c83fd",
            "#06d6a0" if s1_profit >= baseline_profit else "#ef476f",
            "#06d6a0" if s2_profit >= baseline_profit else "#ef476f",
            "#06d6a0" if s3_profit >= baseline_profit else "#ef476f"]

fig, ax = plt.subplots(figsize=(10, 4))
bars = ax.bar(labels, values, color=colors, edgecolor="#2a2d3e", width=0.5)
ax.axhline(baseline_profit, color="#ffd166", linewidth=1.5,
           linestyle="--", label=f"Baseline: ${baseline_profit:,.0f}")
ax.set_ylabel("Predicted Net Profit ($)")
ax.set_title("Scenario Profit Comparison", fontsize=11, pad=10)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.bar_label(bars, labels=[f"${v:,.0f}" for v in values],
             padding=4, color="#e8eaf6", fontsize=9, fontweight="bold")
ax.legend(fontsize=9)
fig.patch.set_facecolor("#161927")
ax.set_facecolor("#161927")
plt.tight_layout()
st.pyplot(fig)
plt.close()

# ── Sensitivity Sweep ─────────────────────────────────────────────────────────
section_header("Commission Rate Sensitivity Sweep")
st.markdown("<p style='color:#9aa0b4; font-size:0.9rem; margin-bottom:0.8rem;'>How does profit change as commission rate increases from 5% to 45%?</p>",
            unsafe_allow_html=True)

comm_range   = np.arange(0.05, 0.46, 0.01)
comm_profits = [predict({**base_params, "comm": c,
                          "comm": c,
                          **{k: base_params[k] for k in base_params if k != "comm"}}) 
                for c in comm_range]

# Fix the sweep properly
sweep_profits = []
for c in comm_range:
    p = {**base_params}
    p["comm"] = c
    p["Commission_UE"] = c * base_params["ue"]
    p["Commission_DD"] = c * base_params["dd"]
    sweep_profits.append(predict(p))

fig2, ax2 = plt.subplots(figsize=(10, 3.5))
ax2.plot(comm_range * 100, sweep_profits, color=PALETTE[0], linewidth=2.5)
ax2.fill_between(comm_range * 100, sweep_profits,
                 alpha=0.15, color=PALETTE[0])
ax2.axvline(base_params["comm"]*100, color=PALETTE[4],
            linestyle="--", linewidth=1.5,
            label=f"Current: {base_params['comm']*100:.1f}%")
ax2.set_xlabel("Commission Rate (%)")
ax2.set_ylabel("Predicted Net Profit ($)")
ax2.set_title("Profit vs Commission Rate", fontsize=11, pad=10)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax2.legend(fontsize=9)
fig2.patch.set_facecolor("#161927")
ax2.set_facecolor("#161927")
plt.tight_layout()
st.pyplot(fig2)
plt.close()

# ── Insight Cards ─────────────────────────────────────────────────────────────
section_header("Strategic Insights")
best_scenario = max([(s1_profit,"Channel Mix Shift"),
                     (s2_profit,"Commission & Cost Change"),
                     (s3_profit,"Growth & Demand Shift")], key=lambda x: x[0])
worst_scenario = min([(s1_profit,"Channel Mix Shift"),
                      (s2_profit,"Commission & Cost Change"),
                      (s3_profit,"Growth & Demand Shift")], key=lambda x: x[0])

col_i1, col_i2 = st.columns(2)
with col_i1:
    insight_card("Best Performing Scenario",
        f"<b>{best_scenario[1]}</b> yields the highest profit at <b>${best_scenario[0]:,.0f}</b> — "
        f"an uplift of <b>${best_scenario[0]-baseline_profit:,.0f}</b> vs baseline.",
        "positive")
with col_i2:
    if worst_scenario[0] < baseline_profit:
        insight_card("Risk Alert",
            f"<b>{worst_scenario[1]}</b> reduces profit to <b>${worst_scenario[0]:,.0f}</b> — "
            f"a <b>${baseline_profit-worst_scenario[0]:,.0f}</b> decline. Review before executing.",
            "risk")
    else:
        insight_card("All Scenarios Profitable",
            f"All three scenarios outperform baseline. Focus on <b>{best_scenario[1]}</b> "
            f"for maximum impact at <b>${best_scenario[0]:,.0f}</b>.",
            "positive")