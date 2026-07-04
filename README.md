# SkyCity Auckland — Restaurant & Bar Profit Intelligence Dashboard

An interactive profit intelligence system for SkyCity Auckland's restaurant and bar portfolio, combining predictive modelling, scenario simulation, and prescriptive optimization into a single Streamlit dashboard.

Built as part of the Data Analyst Internship program at Unified Mentor.

**Live app:** _add deployed Streamlit URL here_
**Research paper:** _add link here_

---

## What this project does

Restaurants selling through multiple channels (in-store, Uber Eats, DoorDash, self-delivery) each carry different cost structures, and small shifts in channel mix or commission rates can move monthly profit by thousands of dollars. This dashboard turns 1,696 restaurant-level records into forward-looking answers:

- **Predicts** total net profit from channel mix and cost inputs using Random Forest, Gradient Boosting, and Linear Regression models
- **Simulates** what-if scenarios (channel mix shifts, commission changes, demand shocks) against a live baseline
- **Optimizes** channel allocation under real business constraints using SLSQP constrained optimization
- **Ranks** cost drivers by profit sensitivity to show where operational attention should go

## Dashboard pages

| Page | Contents |
|---|---|
| Home | Portfolio KPIs, channel performance snapshot, top 10 restaurants by profit |
| Overview | Profit by cuisine/segment, channel distribution, regional performance, cost heatmap |
| Profit Predictor | Model selection, live performance metrics, feature importance, live prediction with confidence range, per-order profit, channel margin breakdown |
| What-If Simulator | Channel mix / commission & cost / growth & demand scenario testing |
| Optimization | SLSQP-based channel-mix optimizer, break-even commission analysis |
| Sensitivity | Profit Sensitivity Index, variable sweep charts, risk & opportunity insights |

## Key findings

- Rebalancing channel mix away from Uber Eats (currently 48.7% of order share) toward in-store sales could lift profit by **45.4%**
- **OPEX rate** is the single largest driver of profit sensitivity — roughly 4x more influential than commission rate
- In-store orders return a **37% margin**, versus roughly **9%** for aggregator-channel orders

## Tech stack

Python · Streamlit · Pandas · NumPy · Scikit-learn · Seaborn · Matplotlib · Plotly · SciPy (SLSQP) · XGBoost · Joblib

## Running locally

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
streamlit run main.py
```

## Project structure

```
skycity_project/
├── main.py                     # Home page
├── app/
│   ├── pages/
│   │   ├── 1_Overview.py
│   │   ├── 2_Profit_Predictor.py
│   │   ├── 3_WhatIf_Simulator.py
│   │   ├── 4_Optimization.py
│   │   └── 5_Sensitivity.py
│   ├── utils/
│   │   ├── data_loader.py
│   │   └── plot_utils.py
│   └── static/
│       └── skycity_logo.jpg
├── data/
│   └── SkyCity_Auckland_Restaurants___Bars.csv
├── .streamlit/
│   └── config.toml
├── requirements.txt
└── README.md
```

## Data

The dataset covers 1,696 restaurant-level records across 8 cuisine types, 4 business segments, and 4 Auckland subregions, with channel-level revenue, cost rates, and profit for in-store, Uber Eats, DoorDash, and self-delivery.

## Author

**Vipsa Patel** — Data Analyst Intern, Unified Mentor
