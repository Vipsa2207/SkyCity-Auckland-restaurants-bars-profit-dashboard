import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.optimize import minimize
import sys, os

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from app.utils.data_loader import load_data, get_cuisines, get_segments, get_subregions
from app.utils.plot_utils import (render_header, render_sidebar_filters, DARK_CSS,
                                   kpi_card, section_header, page_title, insight_card)

st.set_page_config(page_title="Optimization | SkyCity", page_icon="🎯", layout="wide")
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
page_title("Channel Optimization")

# ── Train model ───────────────────────────────────────────────────────────────
@st.cache_data
def train_model(data):
    d = data.copy()
    le = LabelEncoder()
    for col in ["CuisineType", "Segment", "Subregion"]:
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
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=150, random_state=42)
    model.fit(X_tr, y_tr)
    return model, d[features].mean().to_dict(), features

model, base_means, feature_names = train_model(fdf)

# ── Optimization Constraints ──────────────────────────────────────────────────
section_header("Optimization Constraints")
st.markdown("<p style='color:#9aa0b4; font-size:0.9rem; margin-bottom:1rem;'>Set business constraints before running optimization.</p>", unsafe_allow_html=True)

oc1, oc2, oc3, oc4 = st.columns(4)
with oc1: max_comm     = st.slider("Max Commission Rate",       0.10, 0.45, 0.30, 0.01)
with oc2: max_delcost  = st.slider("Max Delivery Cost/Order",   0.5,  6.0,  4.0,  0.1)
with oc3: min_instore  = st.slider("Min In-Store Share",        0.0,  0.5,  0.10, 0.01)
with oc4: min_sd       = st.slider("Min Self-Delivery Share",   0.0,  0.5,  0.10, 0.01)

st.markdown("---")
if min_instore + min_sd > 0.95:
    st.error("⚠️ Min In-Store + Min Self-Delivery exceed 95% — optimization will fail. Please reduce constraints.")
    st.stop()
    
# ── Run Optimization ──────────────────────────────────────────────────────────
def build_input(instore, ue, dd, sd, comm, delcost):
    return np.array([[
        instore, ue, dd, sd,
        comm, delcost,
        base_means["DeliveryRadiusKM"],
        base_means["GrowthFactor"],
        base_means["AOV"],
        base_means["MonthlyOrders"],
        base_means["COGSRate"],
        base_means["OPEXRate"],
        comm * ue, comm * dd, delcost * sd,
        base_means["CuisineType_enc"],
        base_means["Segment_enc"],
        base_means["Subregion_enc"]
    ]])

def neg_profit(x):
    instore, ue, dd, sd = x
    comm    = min(base_means["CommissionRate"], max_comm)
    delcost = min(base_means["DeliveryCostPerOrder"], max_delcost)
    return -model.predict(build_input(instore, ue, dd, sd, comm, delcost))[0]

# Constraints: shares must sum to 1
constraints = [{"type": "eq", "fun": lambda x: x[0]+x[1]+x[2]+x[3]-1.0}]
bounds = [
    (min_instore, 1.0 - min_sd),
    (0.0, 0.8),
    (0.0, 0.8),
    (min_sd, 1.0 - min_instore),
]
x0 = [
    min(max(0.35, min_instore), 0.8),
    0.25,
    0.20,
    min(max(0.20, min_sd), 0.6),
]

with st.spinner("Running optimization..."):
    result = minimize(neg_profit, x0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"maxiter": 500, "ftol": 1e-9})

opt_instore, opt_ue, opt_dd, opt_sd = result.x
opt_profit  = -result.fun
base_profit = model.predict(build_input(
    base_means["InStoreShare"], base_means["UE_share"],
    base_means["DD_share"],     base_means["SD_share"],
    min(base_means["CommissionRate"], max_comm),
    min(base_means["DeliveryCostPerOrder"], max_delcost)
))[0]
uplift      = opt_profit - base_profit
uplift_pct  = (uplift / abs(base_profit)) * 100

# ── Results KPIs ──────────────────────────────────────────────────────────────
section_header("Optimization Results")
k1, k2, k3, k4 = st.columns(4)
with k1: kpi_card("Current Profit",    f"${base_profit:,.0f}", "Baseline average",        "info")
with k2: kpi_card("Optimized Profit",  f"${opt_profit:,.0f}",  "↑ After optimization",    "positive")
with k3: kpi_card("Profit Uplift",     f"${uplift:,.0f}",      f"↑ {uplift_pct:.1f}% gain","positive")
with k4: kpi_card("Optimization",      "✓ Converged" if result.success else "⚠ Check",
                  "SLSQP solver", "positive" if result.success else "negative")

st.markdown("<div style='margin-top:1.2rem;'>", unsafe_allow_html=True)

# ── Optimal Mix vs Current ────────────────────────────────────────────────────
section_header("Optimal Channel Mix vs Current")

col1, col2 = st.columns(2)

with col1:
    channels     = ["In-Store", "Uber Eats", "DoorDash", "Self-Delivery"]
    current_vals = [
        base_means["InStoreShare"], base_means["UE_share"],
        base_means["DD_share"],     base_means["SD_share"]
    ]
    optimal_vals = [opt_instore, opt_ue, opt_dd, opt_sd]

    x      = np.arange(len(channels))
    width  = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars1 = ax.bar(x - width/2, [v*100 for v in current_vals],
                   width, label="Current", color=PALETTE[0], edgecolor="#2a2d3e")
    bars2 = ax.bar(x + width/2, [v*100 for v in optimal_vals],
                   width, label="Optimal", color=PALETTE[2], edgecolor="#2a2d3e")
    ax.set_xticks(x)
    ax.set_xticklabels(channels)
    ax.set_ylabel("Share (%)")
    ax.set_title("Channel Mix: Current vs Optimal", fontsize=11, pad=10)
    ax.bar_label(bars1, labels=[f"{v*100:.1f}%" for v in current_vals],
                 padding=3, color="#e8eaf6", fontsize=8)
    ax.bar_label(bars2, labels=[f"{v*100:.1f}%" for v in optimal_vals],
                 padding=3, color="#e8eaf6", fontsize=8)
    ax.legend(fontsize=9)
    fig.patch.set_facecolor("#161927")
    ax.set_facecolor("#161927")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col2:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    wedge_colors = [PALETTE[0], PALETTE[1], PALETTE[3], PALETTE[2]]
    wedges, texts, autotexts = ax.pie(
        optimal_vals,
        labels=channels,
        autopct="%1.1f%%",
        colors=wedge_colors,
        startangle=140,
        wedgeprops=dict(edgecolor="#0f1117", linewidth=2)
    )
    for at in autotexts:
        at.set_color("#0f1117")
        at.set_fontweight("bold")
        at.set_fontsize(9)
    ax.set_title("Optimal Channel Mix Distribution", fontsize=11, pad=10)
    fig.patch.set_facecolor("#161927")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── Break-Even Commission Analysis ────────────────────────────────────────────
section_header("Break-Even Commission Rate Analysis")
st.markdown("<p style='color:#9aa0b4; font-size:0.9rem; margin-bottom:0.8rem;'>At what commission rate does each channel become unprofitable?</p>",
            unsafe_allow_html=True)

comm_range = np.arange(0.05, 0.50, 0.01)
profits_by_comm = []
for c in comm_range:
    p = model.predict(build_input(
        opt_instore, opt_ue, opt_dd, opt_sd, c,
        base_means["DeliveryCostPerOrder"]
    ))[0]
    profits_by_comm.append(p)

breakeven_idx  = next((i for i, p in enumerate(profits_by_comm) if p <= 0), None)
breakeven_comm = comm_range[breakeven_idx] if breakeven_idx else None

fig3, ax3 = plt.subplots(figsize=(10, 3.5))
ax3.plot(comm_range * 100, profits_by_comm, color=PALETTE[0], linewidth=2.5)
ax3.fill_between(comm_range * 100, profits_by_comm, alpha=0.12, color=PALETTE[0])
ax3.axhline(0, color=PALETTE[3], linewidth=1.5, linestyle="--", label="Break-Even")
ax3.axvline(max_comm * 100, color=PALETTE[4], linewidth=1.5,
            linestyle=":", label=f"Your Max: {max_comm*100:.0f}%")
if breakeven_comm:
    ax3.axvline(breakeven_comm * 100, color=PALETTE[3], linewidth=1.5,
                linestyle="--", label=f"Break-Even: {breakeven_comm*100:.1f}%")
ax3.set_xlabel("Commission Rate (%)")
ax3.set_ylabel("Predicted Net Profit ($)")
ax3.set_title("Profit vs Commission Rate (Optimal Mix)", fontsize=11, pad=10)
ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax3.legend(fontsize=9)
fig3.patch.set_facecolor("#161927")
ax3.set_facecolor("#161927")
plt.tight_layout()
st.pyplot(fig3)
plt.close()

# ── Recommendations Table ─────────────────────────────────────────────────────
section_header("Recommended Channel Allocation")

rec_df = pd.DataFrame({
    "Channel":         channels,
    "Current Share":   [f"{v*100:.1f}%" for v in current_vals],
    "Optimal Share":   [f"{v*100:.1f}%" for v in optimal_vals],
    "Change":          [f"{'↑' if o>c else '↓'} {abs(o-c)*100:.1f}pp"
                        for c,o in zip(current_vals, optimal_vals)],
})
st.dataframe(rec_df, use_container_width=True, hide_index=True)

# ── Insight Cards ─────────────────────────────────────────────────────────────
section_header("Optimization Insights")
col_i1, col_i2 = st.columns(2)

best_channel = channels[np.argmax(optimal_vals)]
worst_channel = channels[np.argmin(optimal_vals)]

with col_i1:
    insight_card("Recommended Priority Channel",
        f"The optimizer recommends maximizing <b>{best_channel}</b> share to "
        f"<b>{max(optimal_vals)*100:.1f}%</b> for peak profitability. "
        f"This delivers a <b>${uplift:,.0f}</b> uplift over current mix.",
        "positive")
with col_i2:
    if breakeven_comm:
        insight_card("Commission Risk Threshold",
            f"Profitability breaks even at <b>{breakeven_comm*100:.1f}%</b> commission. "
            f"Your current cap of <b>{max_comm*100:.0f}%</b> provides a "
            f"<b>{(breakeven_comm-max_comm)*100:.1f}pp</b> safety margin.",
            "positive" if max_comm < breakeven_comm else "risk")
    else:
        insight_card("Commission Risk",
            f"No break-even point found within 5–50% range. "
            f"The optimized mix is highly resilient to commission increases.",
            "positive")