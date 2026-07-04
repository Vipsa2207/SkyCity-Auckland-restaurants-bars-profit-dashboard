import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import sys, os

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from app.utils.data_loader import load_data, get_cuisines, get_segments, get_subregions
from app.utils.plot_utils import (render_header, render_sidebar_filters,
                                   DARK_CSS, kpi_card, section_header,
                                   page_title, insight_card)

st.set_page_config(page_title="Profit Predictor | SkyCity", page_icon="🤖", layout="wide")
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
page_title("Profit Predictor")

# ── Feature Prep ──────────────────────────────────────────────────────────────
@st.cache_data
def prepare_features(data):
    d = data.copy()
    le = LabelEncoder()
    for col in ["CuisineType", "Segment", "Subregion"]:
        d[col + "_enc"] = le.fit_transform(d[col].astype(str))
    features = [
        "InStoreShare", "UE_share", "DD_share", "SD_share",
        "CommissionRate", "DeliveryCostPerOrder", "DeliveryRadiusKM",
        "GrowthFactor", "AOV", "MonthlyOrders", "COGSRate", "OPEXRate",
        "Commission_UE", "Commission_DD", "DeliveryCost_SD",
        "CuisineType_enc", "Segment_enc", "Subregion_enc"
    ]
    target = "TotalNetProfit"
    d = d.dropna(subset=features + [target])
    return d[features], d[target], features

X, y, feature_names = prepare_features(fdf)

# ── Model Config ──────────────────────────────────────────────────────────────
section_header("Model Configuration")
col1, col2, col3 = st.columns(3)
with col1:
    model_choice = st.selectbox("Select Model", [
        "Random Forest", "Gradient Boosting", "Linear Regression"])
with col2:
    test_size = st.slider("Test Set Size (%)", 10, 40, 20, step=5)
with col3:
    n_estimators = st.slider("Number of Trees", 50, 300, 100, step=50)

# ── Train ─────────────────────────────────────────────────────────────────────
@st.cache_data
def train_model(model_name, test_sz, n_est, X_vals, y_vals):
    X_arr = np.array(X_vals)
    y_arr = np.array(y_vals)
    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, test_size=test_sz/100, random_state=42)
    if model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=n_est, random_state=42, n_jobs=-1)
    elif model_name == "Gradient Boosting":
        model = GradientBoostingRegressor(n_estimators=n_est, random_state=42)
    else:
        model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_test, y_pred, X_train, X_test

with st.spinner("Training model..."):
    model, y_test, y_pred, X_train, X_test = train_model(
        model_choice, test_size, n_estimators,
        X.values.tolist(), y.values.tolist())

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)

# ── Model Performance KPIs ────────────────────────────────────────────────────
section_header("Model Performance")
m1, m2, m3, m4 = st.columns(4)
with m1: kpi_card("R² Score",    f"{r2:.4f}",     "Closer to 1.0 is better", "positive" if r2 > 0.7 else "neutral")
with m2: kpi_card("RMSE",        f"${rmse:,.0f}", "Lower is better",          "info")
with m3: kpi_card("MAE",         f"${mae:,.0f}",  "Lower is better",          "info")
with m4: kpi_card("Training Rows", f"{len(X_train):,}", f"{100-test_size}% of data", "info")

st.markdown("<div style='margin-top:1rem;'>", unsafe_allow_html=True)

# ── Auto Insights ─────────────────────────────────────────────────────────────
section_header("Model Insights")
col_a, col_b = st.columns(2)
with col_a:
    if r2 >= 0.85:
        insight_card("Strong Predictive Power",
            f"R² of <b>{r2:.4f}</b> means the model explains {r2*100:.1f}% of profit variance — highly reliable for strategic planning.",
            "positive")
    elif r2 >= 0.65:
        insight_card("Moderate Predictive Power",
            f"R² of <b>{r2:.4f}</b> is acceptable but consider adding more features or trying Gradient Boosting for improvement.",
            "positive")
    else:
        insight_card("Low Predictive Power",
            f"R² of <b>{r2:.4f}</b> suggests high variability. Try switching models or adjusting filters.",
            "risk")
with col_b:
    insight_card("Prediction Accuracy",
        f"On average, predictions are off by <b>${mae:,.0f}</b> — representing about <b>{(mae/y.mean()*100):.1f}%</b> of mean profit.",
        "positive" if mae/y.mean() < 0.2 else "risk")

# ── Charts Row 1 ──────────────────────────────────────────────────────────────
section_header("Actual vs Predicted & Residual Analysis")
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(y_test, y_pred, alpha=0.5, color=PALETTE[0], s=35, edgecolors="none")
    mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], color=PALETTE[3], linewidth=1.8,
            linestyle="--", label="Perfect Fit")
    ax.set_xlabel("Actual Net Profit ($)")
    ax.set_ylabel("Predicted Net Profit ($)")
    ax.set_title(f"Actual vs Predicted — {model_choice}", fontsize=11, pad=10)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=9)
    fig.patch.set_facecolor("#161927")
    ax.set_facecolor("#161927")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

with col2:
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.histplot(residuals, bins=40, color=PALETTE[0], ax=ax, kde=True,
                 line_kws={"color": PALETTE[4], "linewidth": 2})
    ax.axvline(0, color=PALETTE[3], linestyle="--", linewidth=1.5, label="Zero Error")
    ax.set_xlabel("Residual ($)")
    ax.set_title("Residual Distribution", fontsize=11, pad=10)
    ax.legend(fontsize=9)
    fig.patch.set_facecolor("#161927")
    ax.set_facecolor("#161927")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

# ── Feature Importance ────────────────────────────────────────────────────────
section_header("Feature Importance")

if hasattr(model, "feature_importances_"):
    imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=True).tail(12)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.barh(imp_df["Feature"], imp_df["Importance"],
                   color=PALETTE[0], edgecolor="#2a2d3e", height=0.6)
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Top Feature Importances — {model_choice}", fontsize=11, pad=10)
    ax.bar_label(bars, labels=[f"{v:.4f}" for v in imp_df["Importance"]],
                 padding=4, color="#e8eaf6", fontsize=8)
    fig.patch.set_facecolor("#161927")
    ax.set_facecolor("#161927")
    plt.tight_layout()
    st.pyplot(fig); plt.close()
else:
    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": np.abs(model.coef_)
    }).sort_values("Coefficient", ascending=True).tail(12)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.barh(coef_df["Feature"], coef_df["Coefficient"],
            color=PALETTE[1], edgecolor="#2a2d3e", height=0.6)
    ax.set_xlabel("|Coefficient|")
    ax.set_title("Feature Coefficients — Linear Regression", fontsize=11, pad=10)
    fig.patch.set_facecolor("#161927")
    ax.set_facecolor("#161927")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

# ── Live Predictor ────────────────────────────────────────────────────────────
section_header("Live Profit Predictor")
st.markdown("<p style='color:#9aa0b4; font-size:0.9rem; margin-bottom:1rem;'>Adjust inputs below to get an instant profit prediction from the trained model.</p>", unsafe_allow_html=True)

lc1, lc2, lc3 = st.columns(3)

with lc1:
    st.markdown("**Channel Mix**")
    p_instore = st.slider("In-Store Share",      0.0, 1.0, 0.35, 0.01)
    p_ue      = st.slider("Uber Eats Share",     0.0, 1.0, 0.25, 0.01)
    p_dd      = st.slider("DoorDash Share",      0.0, 1.0, 0.20, 0.01)
    p_sd      = st.slider("Self-Delivery Share", 0.0, 1.0, 0.20, 0.01)
    p_aov     = st.slider("Avg Order Value ($)", 25.0, 55.0, 38.0, 0.5)
    p_orders  = st.slider("Monthly Orders",      200, 3000, 1000, 50)

with lc2:
    st.markdown("**Cost Variables**")
    p_comm    = st.slider("Commission Rate",         0.10, 0.45, 0.28, 0.01)
    p_delcost = st.slider("Delivery Cost/Order ($)", 0.5, 6.0, 3.0, 0.1)
    p_radius  = st.slider("Delivery Radius (km)",    2, 20, 10, 1)
    p_growth  = st.slider("Growth Factor",           0.95, 1.10, 1.02, 0.01)

with lc3:
    st.markdown("**Cost Rates & Profile**")
    p_cogs      = st.slider("COGS Rate",  0.15, 0.45, 0.25, 0.01)
    p_opex      = st.slider("OPEX Rate",  0.20, 0.60, 0.38, 0.01)
    p_cuisine   = st.selectbox("Cuisine Type", ["Burgers","Pizza","Indian","Chinese",
                                                 "Japanese","Thai","Chicken Dishes","Kebabs/Mediterranean"])
    p_segment   = st.selectbox("Segment", ["Cafe","QSR","Full-service","Ghost Kitchen"])
    p_subregion = st.selectbox("Subregion", ["North Shore","South Auckland","West Auckland","CBD"])

cuisine_map   = {"Burgers":0,"Pizza":1,"Indian":2,"Chinese":3,"Japanese":4,"Thai":5,"Chicken Dishes":6,"Kebabs/Mediterranean":7}
segment_map   = {"Cafe":0,"QSR":1,"Full-service":2,"Ghost Kitchen":3}
subregion_map = {"North Shore":0,"South Auckland":1,"West Auckland":2,"CBD":3}

input_vec = np.array([[
    p_instore, p_ue, p_dd, p_sd,
    p_comm, p_delcost, p_radius, p_growth, p_aov, p_orders,
    p_cogs, p_opex,
    p_comm * p_ue, p_comm * p_dd, p_delcost * p_sd,
    cuisine_map[p_cuisine], segment_map[p_segment], subregion_map[p_subregion]
]])

predicted = model.predict(input_vec)[0]
total_share = p_instore + p_ue + p_dd + p_sd
if abs(total_share - 1.0) > 0.05:
    st.warning(f"⚠️ Channel shares sum to {total_share:.2f} — they should sum to 1.0 for accurate predictions.")

# ── Confidence Band (from residual distribution) ─────────────────────────
residual_std = np.std(y_test - y_pred)
lower_90 = predicted - 1.645 * residual_std
upper_90 = predicted + 1.645 * residual_std

# ── Net Profit per Order ──────────────────────────────────────────────────
profit_per_order = predicted / p_orders if p_orders > 0 else 0

# ── Channel-Level Margin Breakdown ────────────────────────────────────────
rev_instore = p_instore * p_orders * p_aov
rev_ue      = p_ue      * p_orders * p_aov
rev_dd      = p_dd      * p_orders * p_aov
rev_sd      = p_sd      * p_orders * p_aov

profit_instore = rev_instore * (1 - p_cogs - p_opex)
profit_ue      = rev_ue * (1 - p_cogs - p_opex - p_comm)
profit_dd      = rev_dd * (1 - p_cogs - p_opex - p_comm)
profit_sd      = rev_sd * (1 - p_cogs - p_opex) - (p_delcost * p_sd * p_orders)

margin_instore = (profit_instore / rev_instore * 100) if rev_instore > 0 else 0
margin_ue      = (profit_ue / rev_ue * 100) if rev_ue > 0 else 0
margin_dd      = (profit_dd / rev_dd * 100) if rev_dd > 0 else 0
margin_sd      = (profit_sd / rev_sd * 100) if rev_sd > 0 else 0


color = "#06d6a0" if predicted >= 0 else "#ef476f"
label = "Profitable" if predicted >= 0 else "Loss-Making"

st.markdown("---")
r1, r2_col, r3 = st.columns([1, 2, 1])
with r2_col:
    st.markdown(f"""
    <div style='background:#161927; border:2px solid {color}; border-radius:14px;
                padding:28px 24px; text-align:center;'>
        <p style='color:#9aa0b4; font-size:0.85rem; margin:0 0 6px 0; 
                  text-transform:uppercase; letter-spacing:0.08em;'>
            Predicted Monthly Net Profit
        </p>
        <h1 style='color:{color}; font-size:2.8rem; margin:0; font-weight:800;'>
            ${predicted:,.0f}
        </h1>
        <p style='color:#9aa0b4; font-size:0.8rem; margin:6px 0 0 0;'>
            90% confidence range: ${lower_90:,.0f} – ${upper_90:,.0f}
        </p>
        <p style='color:{color}; font-size:0.85rem; margin:8px 0 0 0; font-weight:600;'>
            {label} · {model_choice} · R² = {r2:.4f}
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-top: 1.8rem;'></div>", unsafe_allow_html=True)

# ── Net Profit per Order & Confidence KPIs ───────────────────────────────────
section_header("Per-Order & Channel-Level Insights")
k1, k2 = st.columns(2)
with k1:
    kpi_card("Net Profit per Order", f"${profit_per_order:,.2f}", "Predicted profit ÷ orders", "info")
with k2:
    kpi_card("Prediction Range", f"±${1.645*residual_std:,.0f}", "90% confidence band", "neutral")

st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

# ── Channel-Level Margin Table ───────────────────────────────────────────────
st.markdown("<p style='color:#9aa0b4; font-size:0.85rem; font-weight:600; margin-bottom:0.5rem;'>Channel-Level Margin Breakdown</p>", unsafe_allow_html=True)
channel_margin_df = pd.DataFrame({
    "Channel": ["In-Store", "Uber Eats", "DoorDash", "Self-Delivery"],
    "Revenue": [rev_instore, rev_ue, rev_dd, rev_sd],
    "Net Profit": [profit_instore, profit_ue, profit_dd, profit_sd],
    "Margin (%)": [margin_instore, margin_ue, margin_dd, margin_sd]
})
st.dataframe(
    channel_margin_df.style.format({"Revenue": "${:,.0f}", "Net Profit": "${:,.0f}", "Margin (%)": "{:.1f}%"}),
    use_container_width=True, hide_index=True
)