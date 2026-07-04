import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from app.utils.data_loader import load_data, get_cuisines, get_segments, get_subregions
from app.utils.plot_utils import render_header, render_sidebar_filters, DARK_CSS, kpi_card, section_header, page_title

st.set_page_config(page_title="Overview | SkyCity", page_icon="📊", layout="wide")
st.markdown(DARK_CSS, unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor": "#0f1117", "axes.facecolor": "#1e2130",
    "axes.edgecolor": "#3a3d4e",   "axes.labelcolor": "#e8eaf6",
    "xtick.color": "#9e9e9e",      "ytick.color": "#9e9e9e",
    "text.color": "#e8eaf6",       "grid.color": "#2a2d3e",
    "grid.linestyle": "--",        "grid.alpha": 0.5,
})
PALETTE = ["#7c83fd","#f77f00","#06d6a0","#ef476f","#ffd166"]

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
page_title("Business Overview & KPI Trends")


# ── KPI Row ───────────────────────────────────────────────────────────────────
from app.utils.plot_utils import kpi_card, section_header

k1,k2,k3,k4,k5 = st.columns(5)
with k1: kpi_card("Total Revenue",    f"${fdf['TotalRevenue'].sum():,.0f}", "↑ Gross", "positive")
with k2: kpi_card("Total Net Profit", f"${fdf['TotalNetProfit'].sum():,.0f}", "↑ After costs", "positive")
with k3: kpi_card("Avg Profit Margin",f"{fdf['ProfitMargin'].mean():.1f}%", "Portfolio avg", "neutral")
with k4: kpi_card("Avg Order Value",  f"${fdf['AOV'].mean():.2f}", "Per transaction", "info")
with k5: kpi_card("Total Restaurants",f"{fdf['RestaurantID'].nunique():,}", "In selection", "info")

st.markdown("---")

# ── Row 1: Profit by Cuisine & Segment ───────────────────────────────────────
st.markdown("### 🍽️ Profit by Cuisine Type & Segment")
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(7, 4))
    cuisine_profit = fdf.groupby("CuisineType")["TotalNetProfit"].sum().sort_values(ascending=True)
    bars = ax.barh(cuisine_profit.index, cuisine_profit.values, color=PALETTE[0], edgecolor="#3a3d4e")
    ax.set_xlabel("Total Net Profit ($)")
    ax.set_title("Net Profit by Cuisine Type")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.bar_label(bars, labels=[f"${v:,.0f}" for v in cuisine_profit.values],
                 padding=4, color="#e8eaf6", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col2:
    fig, ax = plt.subplots(figsize=(7, 4))
    seg_profit = fdf.groupby("Segment")["TotalNetProfit"].sum().sort_values(ascending=True)
    bars = ax.barh(seg_profit.index, seg_profit.values, color=PALETTE[1], edgecolor="#3a3d4e")
    ax.set_xlabel("Total Net Profit ($)")
    ax.set_title("Net Profit by Segment")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.bar_label(bars, labels=[f"${v:,.0f}" for v in seg_profit.values],
                 padding=4, color="#e8eaf6", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# ── Row 2: Channel Revenue Pie & Subregion Bar ────────────────────────────────
st.markdown("### 📡 Channel Revenue Distribution & Regional Performance")
col3, col4 = st.columns(2)

with col3:
    fig, ax = plt.subplots(figsize=(6, 5))
    channel_rev = {
        "In-Store":      fdf["InStoreRevenue"].sum(),
        "Uber Eats":     fdf["UberEatsRevenue"].sum(),
        "DoorDash":      fdf["DoorDashRevenue"].sum(),
        "Self-Delivery": fdf["SelfDeliveryRevenue"].sum(),
    }
    wedges, texts, autotexts = ax.pie(
        channel_rev.values(),
        labels=channel_rev.keys(),
        autopct="%1.1f%%",
        colors=PALETTE,
        startangle=140,
        wedgeprops=dict(edgecolor="#0f1117", linewidth=2)
    )
    for at in autotexts:
        at.set_color("#0f1117")
        at.set_fontweight("bold")
    ax.set_title("Revenue Share by Channel")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col4:
    fig, ax = plt.subplots(figsize=(7, 5))
    region_profit = fdf.groupby("Subregion")["TotalNetProfit"].sum().sort_values(ascending=True)
    bars = ax.barh(region_profit.index, region_profit.values, color=PALETTE[2], edgecolor="#3a3d4e")
    ax.set_xlabel("Total Net Profit ($)")
    ax.set_title("Net Profit by Subregion")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.bar_label(bars, labels=[f"${v:,.0f}" for v in region_profit.values],
                 padding=4, color="#e8eaf6", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# ── Row 3: Profit Margin Distribution & AOV vs Profit ─────────────────────────
st.markdown("### 📈 Margin Distribution & Order Value Analysis")
col5, col6 = st.columns(2)

with col5:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(fdf["ProfitMargin"], bins=40, color=PALETTE[0], ax=ax, kde=True,
                 line_kws={"color": PALETTE[4], "linewidth": 2})
    ax.set_xlabel("Profit Margin (%)")
    ax.set_title("Distribution of Profit Margins")
    ax.axvline(fdf["ProfitMargin"].mean(), color=PALETTE[3], linestyle="--",
               linewidth=1.5, label=f"Mean: {fdf['ProfitMargin'].mean():.1f}%")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col6:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.scatterplot(data=fdf, x="AOV", y="TotalNetProfit", hue="CuisineType",
                    palette=PALETTE, ax=ax, alpha=0.7, s=60)
    ax.set_xlabel("Average Order Value ($)")
    ax.set_ylabel("Total Net Profit ($)")
    ax.set_title("AOV vs Net Profit by Cuisine")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# ── Row 4: Cost Structure Heatmap ─────────────────────────────────────────────
st.markdown("### 🔥 Cost Structure Heatmap by Cuisine & Segment")

fig, ax = plt.subplots(figsize=(12, 4))
heat_data = fdf.groupby(["CuisineType", "Segment"])[["COGSRate","OPEXRate","CommissionRate"]].mean().unstack()
sns.heatmap(heat_data, annot=True, fmt=".2f", cmap="YlOrRd",
            ax=ax, linewidths=0.5, linecolor="#0f1117",
            cbar_kws={"shrink": 0.8})
ax.set_title("Average Cost Rates (COGS / OPEX / Commission) by Cuisine × Segment")
ax.set_xlabel("")
plt.tight_layout()
st.pyplot(fig)
plt.close()