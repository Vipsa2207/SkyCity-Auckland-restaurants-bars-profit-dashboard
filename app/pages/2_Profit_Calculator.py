import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from app.utils.data_loader import load_data, get_cuisines, get_segments, get_subregions
from app.utils.plot_utils import (render_header, render_sidebar_filters, DARK_CSS,
                                   kpi_card, section_header, page_title, insight_card)
from app.utils import profit_engine as pe

st.set_page_config(page_title="Profit Calculator | SkyCity", page_icon="🧮", layout="wide")
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
page_title("Profit Calculator",
           "Profit in this dataset follows a fixed formula, so we calculate it exactly instead of predicting it.")

if fdf.empty:
    st.warning("No restaurants match the current filters.")
    st.stop()

# ── How profit is built ──────────────────────────────────────────────────────
section_header("How Profit Is Calculated")
st.markdown("""
<div class='chart-card' style='font-size:0.92rem; line-height:1.7; color:#c9cce0;'>
<b>Revenue</b> (each channel) = orders × average order value<br>
<b>In-Store profit</b> = revenue × (1 − food cost rate − operating cost rate)<br>
<b>Uber Eats / DoorDash profit</b> = revenue × (1 − food cost rate − operating cost rate − <b>commission</b>)<br>
<b>Self-Delivery profit</b> = revenue × (1 − food cost rate − operating cost rate) − orders × <b>delivery cost per order</b>
</div>
""", unsafe_allow_html=True)
st.markdown("<div style='margin-top:0.8rem;'></div>", unsafe_allow_html=True)

gap = pe.formula_check(fdf)
c1, c2, c3 = st.columns(3)
with c1: kpi_card("Restaurants checked", f"{len(fdf):,}", "Every row in the selection", "info")
with c2: kpi_card("Largest gap vs dataset", f"${gap:,.2f}", "Formula matches the data", "positive")
with c3: kpi_card("Avg commission", f"{fdf['CommissionRate'].mean()*100:.1f}%", "Uber Eats & DoorDash", "neutral")
st.caption("Why no machine-learning model? An earlier version trained Random Forest / Gradient Boosting to "
           "predict profit and got a near-perfect score. That score was meaningless: the model was re-learning this "
           "formula from columns built out of the same numbers (data leakage). Calculating directly is exact and explainable.")

# ── Channel economics for the selection ──────────────────────────────────────
section_header("Channel Economics (current selection)")
t = pe.run_scenario(fdf)
tot_orders, tot_profit = t["Orders"].sum(), t["Net Profit"].sum()
t["Order Share (%)"] = t["Orders"] / tot_orders * 100
t["Profit Share (%)"] = t["Net Profit"] / tot_profit * 100 if tot_profit != 0 else 0.0

show = t[["Channel", "Orders", "Order Share (%)", "Revenue", "Net Profit", "Profit Share (%)",
          "Margin (%)", "Profit / Order"]]
st.dataframe(show.style.format({
    "Orders": "{:,.0f}", "Order Share (%)": "{:.1f}%", "Revenue": "${:,.0f}",
    "Net Profit": "${:,.0f}", "Profit Share (%)": "{:.1f}%", "Margin (%)": "{:.1f}%",
    "Profit / Order": "${:,.2f}"}), width="stretch", hide_index=True)

col1, col2 = st.columns([1.2, 1])
with col1:
    fig, ax = plt.subplots(figsize=(7, 3.4))
    y = range(len(t))
    ax.barh([i - 0.2 for i in y], t["Order Share (%)"], height=0.38, color="#5b6075", label="Share of orders")
    ax.barh([i + 0.2 for i in y], t["Profit Share (%)"], height=0.38,
            color=[CH_COLORS[c] for c in t["Channel"]], label="Share of profit (coloured)")
    ax.set_yticks(list(y)); ax.set_yticklabels(t["Channel"]); ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_title("Share of orders (grey) vs share of profit (colour)", fontsize=11)
    ax.grid(axis="x")
    plt.tight_layout(); st.pyplot(fig); plt.close()

agg = t[t["Channel"].isin(pe.AGGREGATORS)]
ins = t[t["Channel"] == "In-Store"].iloc[0]
ue = t[t["Channel"] == "Uber Eats"].iloc[0]
with col2:
    insight_card("Delivery apps: lots of orders, little profit",
        f"Uber Eats and DoorDash carry <b>{agg['Order Share (%)'].sum():.0f}%</b> of orders but only "
        f"<b>{agg['Profit Share (%)'].sum():.0f}%</b> of profit in this selection.", "risk")
    insight_card("Profit per order",
        f"An in-store order earns <b>${ins['Profit / Order']:.2f}</b>; an Uber Eats order earns "
        f"<b>${ue['Profit / Order']:.2f}</b>. Commission (~{fdf['CommissionRate'].mean()*100:.0f}%) eats almost "
        f"all of the margin left after food and operating costs.", "positive")

# ── Single-restaurant calculator ─────────────────────────────────────────────
section_header("Single-Restaurant Calculator")
st.caption("Set up one restaurant and see its exact monthly profit by channel. Defaults are the selection's averages.")

m = fdf
lc1, lc2, lc3 = st.columns(3)
with lc1:
    st.markdown("**Orders per month, by channel**")
    o_in = st.number_input("In-Store orders", 0, 5000, int(m["InStoreOrders"].mean()), 10)
    o_ue = st.number_input("Uber Eats orders", 0, 5000, int(m["UberEatsOrders"].mean()), 10)
    o_dd = st.number_input("DoorDash orders", 0, 5000, int(m["DoorDashOrders"].mean()), 10)
    o_sd = st.number_input("Self-Delivery orders", 0, 5000, int(m["SelfDeliveryOrders"].mean()), 10)
with lc2:
    st.markdown("**Prices & delivery**")
    aov = st.slider("Average order value ($)", 20.0, 60.0, float(round(m["AOV"].mean(), 1)), 0.5)
    comm = st.slider("Commission rate (%)", 10, 45, int(round(m["CommissionRate"].mean() * 100)), 1)
    dcost = st.slider("Delivery cost per order ($)", 0.5, 8.0, float(round(m["DeliveryCostPerOrder"].mean(), 1)), 0.1)
with lc3:
    st.markdown("**Cost rates (share of revenue)**")
    cogs = st.slider("Food cost rate / COGS (%)", 10, 50, int(round(m["COGSRate"].mean() * 100)), 1)
    opex = st.slider("Operating cost rate / OPEX (%)", 15, 70, int(round(m["OPEXRate"].mean() * 100)), 1)
    st.caption(f"Food + operating costs = {cogs + opex}% of revenue before any commission.")

orders = {"In-Store": o_in, "Uber Eats": o_ue, "DoorDash": o_dd, "Self-Delivery": o_sd}
res = pe.channel_profit(orders, aov, cogs / 100, opex / 100, comm / 100, dcost)
calc = pd.DataFrame([{"Channel": ch, "Orders": orders[ch], "Revenue": res[ch][0], "Net Profit": res[ch][1]}
                     for ch in pe.CHANNELS])
calc["Margin (%)"] = [p / r * 100 if r else 0 for r, p in zip(calc["Revenue"], calc["Net Profit"])]
calc["Profit / Order"] = [p / o if o else 0 for o, p in zip(calc["Orders"], calc["Net Profit"])]
total = calc["Net Profit"].sum()
total_orders = calc["Orders"].sum()

color = "#06d6a0" if total >= 0 else "#ef476f"
r1, r2, r3 = st.columns([1, 2, 1])
with r2:
    st.markdown(f"""
    <div style='background:#161927; border:2px solid {color}; border-radius:14px; padding:24px; text-align:center;'>
        <p style='color:#9aa0b4; font-size:0.85rem; margin:0 0 6px 0; text-transform:uppercase; letter-spacing:0.08em;'>
            Monthly Net Profit (exact)</p>
        <h1 style='color:{color}; font-size:2.6rem; margin:0; font-weight:800;'>${total:,.0f}</h1>
        <p style='color:#9aa0b4; font-size:0.85rem; margin:6px 0 0 0;'>
            {total_orders:,} orders · ${total / total_orders if total_orders else 0:,.2f} profit per order</p>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
st.dataframe(calc.style.format({"Orders": "{:,.0f}", "Revenue": "${:,.0f}", "Net Profit": "${:,.0f}",
                                "Margin (%)": "{:.1f}%", "Profit / Order": "${:,.2f}"}),
             width="stretch", hide_index=True)
