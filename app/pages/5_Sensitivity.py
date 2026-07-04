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

st.set_page_config(page_title="Sensitivity | SkyCity", page_icon="📉", layout="wide")
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
page_title("Sensitivity Analysis")

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
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=150, random_state=42)
    model.fit(X_tr, y_tr)
    return model, d[features].mean().to_dict(), features

model, base_means, feature_names = train_model(fdf)

def predict_from_dict(params):
    vec = np.array([[params[f] for f in feature_names]])
    return model.predict(vec)[0]

base_profit = predict_from_dict(base_means)

# ── Sensitivity Config ────────────────────────────────────────────────────────
section_header("Sensitivity Configuration")
st.markdown("<p style='color:#9aa0b4; font-size:0.9rem; margin-bottom:1rem;'>Choose how much to vary each variable (as % of its current value).</p>",
            unsafe_allow_html=True)

sc1, sc2 = st.columns(2)
with sc1:
    shock_pct = st.slider("Shock Size (% change per step)", 5, 30, 10, 5)
with sc2:
    n_steps = st.slider("Number of Steps", 3, 10, 5)

st.markdown("---")

# ── Compute Sensitivities ─────────────────────────────────────────────────────
sensitivity_vars = {
    "CommissionRate":      "Commission Rate",
    "COGSRate":            "COGS Rate",
    "OPEXRate":            "OPEX Rate",
    "DeliveryCostPerOrder":"Delivery Cost/Order",
    "AOV":                 "Avg Order Value",
    "MonthlyOrders":       "Monthly Orders",
    "GrowthFactor":        "Growth Factor",
    "DeliveryRadiusKM":    "Delivery Radius",
}

@st.cache_data
def compute_sensitivity(base, shock, steps, feat_names):
    results = {}
    for var, label in sensitivity_vars.items():
        if var not in base:
            continue
        base_val = base[var]
        deltas, profits = [], []
        for s in range(-steps, steps+1):
            pct    = 1 + (s * shock / 100)
            new_val = base_val * pct
            params  = dict(base)
            params[var] = new_val
            # update interaction terms
            if var == "CommissionRate":
                params["Commission_UE"] = new_val * base["UE_share"]
                params["Commission_DD"] = new_val * base["DD_share"]
            if var == "DeliveryCostPerOrder":
                params["DeliveryCost_SD"] = new_val * base["SD_share"]
            deltas.append(s * shock)
            profits.append(predict_from_dict(params))
        results[label] = {"deltas": deltas, "profits": profits, "base_val": base_val}
    return results

sens_results = compute_sensitivity(
    base_means, shock_pct, n_steps, feature_names)

# ── Sensitivity Index ─────────────────────────────────────────────────────────
section_header("Profit Sensitivity Index")
st.markdown("<p style='color:#9aa0b4; font-size:0.9rem; margin-bottom:1rem;'>How much does a 1% change in each variable shift net profit? Higher = more sensitive.</p>",
            unsafe_allow_html=True)

sens_index = {}
for label, data in sens_results.items():
    profits = np.array(data["profits"])
    deltas  = np.array(data["deltas"])
    # slope: $ change per 1% shock
    slope = np.polyfit(deltas, profits, 1)[0]
    sens_index[label] = abs(slope)

sens_df = pd.DataFrame({
    "Variable":        list(sens_index.keys()),
    "Sensitivity ($/$1% change)": list(sens_index.values())
}).sort_values("Sensitivity ($/$1% change)", ascending=True)

fig, ax = plt.subplots(figsize=(10, 4))
colors = [PALETTE[3] if v > np.median(list(sens_index.values()))
          else PALETTE[0] for v in sens_df["Sensitivity ($/$1% change)"]]
bars = ax.barh(sens_df["Variable"], sens_df["Sensitivity ($/$1% change)"],
               color=colors, edgecolor="#2a2d3e", height=0.6)
ax.set_xlabel("$ Profit Change per 1% Variable Change")
ax.set_title("Profit Sensitivity Index — All Variables", fontsize=11, pad=10)
ax.bar_label(bars, labels=[f"${v:,.0f}" for v in sens_df["Sensitivity ($/$1% change)"]],
             padding=4, color="#e8eaf6", fontsize=8)
fig.patch.set_facecolor("#161927")
ax.set_facecolor("#161927")
plt.tight_layout()
st.pyplot(fig)
plt.close()

# ── Top Variable KPIs ─────────────────────────────────────────────────────────
top3 = sens_df.tail(3)[::-1].reset_index(drop=True)
k1, k2, k3 = st.columns(3)
cols = [k1, k2, k3]
risk_colors = ["negative", "neutral", "info"]
for i, row in top3.iterrows():
    with cols[i]:
        kpi_card(f"#{i+1} Most Sensitive",
                 row["Variable"],
                 f"${row['Sensitivity ($/$1% change)']:,.0f} per 1% change",
                 risk_colors[i])

st.markdown("<div style='margin-top:1.2rem;'>", unsafe_allow_html=True)

# ── Individual Sweep Charts ───────────────────────────────────────────────────
section_header("Variable Sweep Analysis")
st.markdown("<p style='color:#9aa0b4; font-size:0.9rem; margin-bottom:1rem;'>Profit response curve for each variable independently swept ±{shock_pct}%.</p>".replace("{shock_pct}", str(shock_pct*n_steps)),
            unsafe_allow_html=True)

var_labels = list(sens_results.keys())
n_vars = len(var_labels)
n_cols = 2
n_rows = (n_vars + 1) // n_cols

for row_i in range(n_rows):
    cols = st.columns(n_cols)
    for col_i in range(n_cols):
        idx = row_i * n_cols + col_i
        if idx >= n_vars:
            break
        label = var_labels[idx]
        data  = sens_results[label]
        with cols[col_i]:
            fig, ax = plt.subplots(figsize=(6, 3))
            color = PALETTE[3] if sens_index[label] > np.median(list(sens_index.values())) else PALETTE[0]
            ax.plot(data["deltas"], data["profits"], color=color,
                    linewidth=2.2, marker="o", markersize=4)
            ax.fill_between(data["deltas"], data["profits"],
                            base_profit, alpha=0.1, color=color)
            ax.axhline(base_profit, color="#ffd166", linewidth=1.2,
                       linestyle="--", label=f"Baseline: ${base_profit:,.0f}")
            ax.axvline(0, color="#6b7280", linewidth=0.8, linestyle=":")
            ax.set_xlabel(f"% Change in {label}")
            ax.set_ylabel("Net Profit ($)")
            ax.set_title(label, fontsize=10, pad=6)
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
            ax.legend(fontsize=7)
            fig.patch.set_facecolor("#161927")
            ax.set_facecolor("#161927")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# ── Insight Cards ─────────────────────────────────────────────────────────────
section_header("Risk & Opportunity Insights")

most_sensitive  = sens_df.iloc[-1]
least_sensitive = sens_df.iloc[0]

col_i1, col_i2 = st.columns(2)
with col_i1:
    insight_card("Highest Risk Variable",
        f"<b>{most_sensitive['Variable']}</b> is the most profit-sensitive variable — "
        f"a 1% change shifts profit by <b>${most_sensitive['Sensitivity ($/$1% change)']:,.0f}</b>. "
        f"Prioritise controlling this variable in negotiations and operations.",
        "risk")
with col_i2:
    insight_card("Most Stable Variable",
        f"<b>{least_sensitive['Variable']}</b> has the lowest sensitivity at "
        f"<b>${least_sensitive['Sensitivity ($/$1% change)']:,.0f}</b> per 1% change — "
        f"changes here have minimal profit impact and are safe to adjust.",
        "positive")