import streamlit as st
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.utils.data_loader import load_data, get_cuisines, get_segments, get_subregions
from app.utils.plot_utils import render_header, render_sidebar_filters, DARK_CSS, kpi_card, section_header

st.set_page_config(page_title="SkyCity Auckland | Analytics", page_icon="🏙️", layout="wide")
st.markdown(DARK_CSS, unsafe_allow_html=True)

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

# ── KPI Row ───────────────────────────────────────────────────────────────
c1,c2,c3,c4,c5,c6 = st.columns(6)
with c1: kpi_card("Restaurants",     f"{fdf['RestaurantID'].nunique():,}", "In selection", "info")
with c2: kpi_card("Monthly Orders",  f"{fdf['MonthlyOrders'].sum():,.0f}", "Across all channels", "info")
with c3: kpi_card("Total Revenue",   f"${fdf['TotalRevenue'].sum()/1e6:,.1f}M", "↑ Gross revenue", "positive")
with c4: kpi_card("Net Profit",      f"${fdf['TotalNetProfit'].sum()/1e6:,.2f}M", "↑ After all costs", "positive")
with c5: kpi_card("Avg Margin",      f"{fdf['ProfitMargin'].mean():.1f}%", "Portfolio average", "neutral")
with c6: kpi_card("Avg Order Value", f"${fdf['AOV'].mean():.2f}", "Per transaction", "info")

# ── Channel Snapshot ──────────────────────────────────────────────────────
section_header("Channel Performance Snapshot")
d1,d2,d3,d4 = st.columns(4)
with d1: kpi_card("In-Store Profit",      f"${fdf['InStoreNetProfit'].sum():,.0f}", "🏠 Highest control", "positive")
with d2: kpi_card("Uber Eats Profit",     f"${fdf['UberEatsNetProfit'].sum():,.0f}", "🟢 High commission", "neutral")
with d3: kpi_card("DoorDash Profit",      f"${fdf['DoorDashNetProfit'].sum():,.0f}", "🔴 Smallest share", "neutral")
with d4: kpi_card("Self-Delivery Profit", f"${fdf['SelfDeliveryNetProfit'].sum():,.0f}", "🚗 Best margin", "positive")

# ── Top 10 Table ──────────────────────────────────────────────────────────
section_header("Top 10 Restaurants by Net Profit")
top10 = (
    fdf.groupby(["RestaurantName","CuisineType","Segment","Subregion"])
    [["TotalRevenue","TotalNetProfit","ProfitMargin","MonthlyOrders"]]
    .mean().sort_values("TotalNetProfit", ascending=False)
    .head(10).reset_index()
)
top10["TotalRevenue"]   = top10["TotalRevenue"].map("${:,.0f}".format)
top10["TotalNetProfit"] = top10["TotalNetProfit"].map("${:,.0f}".format)
top10["ProfitMargin"]   = top10["ProfitMargin"].map("{:.1f}%".format)
top10["MonthlyOrders"]  = top10["MonthlyOrders"].map("{:,.0f}".format)
st.dataframe(top10, use_container_width=True, hide_index=True)