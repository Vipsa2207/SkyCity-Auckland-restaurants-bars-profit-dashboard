# SkyCity Auckland — Restaurant & Bar Profit Dashboard

**Live app:** https://skycity-auckland-restaurants-bars-profit-dashboard-7ahgbe8axnd.streamlit.app/

A Streamlit dashboard that shows where 1,696 Auckland restaurants make and lose profit across four sales channels: in-store, Uber Eats, DoorDash and self-delivery.

Project 4 of 4 · Built during my Data Analyst Internship at Unified Mentor (Jan–Jul 2026).

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)

![Dashboard](docs/dashboard.png)

## The business question

Restaurants sell through several channels, each with different costs. Which channels actually make money, and where should the portfolio focus?

## Key findings

- **Delivery apps bring volume, not profit.** Uber Eats and DoorDash carry **61%** of orders but earn only **5%** of net profit. Their net margin is under 1%.
- **Per order, the gap is stark.** An in-store order earns **$10.31** profit and a self-delivered order **$8.82**. An Uber Eats order earns **$0.32**.
- **Costs move profit more than commission.** One percentage point of operating or food cost changes monthly profit by **$777K**. One point of commission changes it by **$476K**. That's about **1.6×**, because costs apply to every order and commission only to app orders.
- **All 408 loss-making restaurants are Full-service.** Their food and operating costs take **87.5%** of revenue before any commission, so the segment loses money overall (**−5.8%** margin). Ghost kitchens earn the most (**22.2%**).
- **538 of 1,696 restaurants lose money on every delivery-app order**, because food cost + operating cost + commission is more than 100% of the order value.
- **Moving orders off the apps has real value.** Moving 10% of Uber Eats orders to the restaurants' own delivery would add up to **$676K a month (+8.6%)**, if those customers keep ordering.

Across the portfolio: $77.7M monthly revenue, $7.87M net profit, a 10.1% margin.

## How it works

1. **Check the data.** Every profit column follows a fixed formula. I checked it on all 1,696 rows, and the largest gap is half a cent:
   - revenue = orders × average order value
   - in-store profit = revenue × (1 − food cost rate − operating cost rate)
   - Uber Eats / DoorDash profit = revenue × (1 − food cost rate − operating cost rate − commission)
   - self-delivery profit = revenue × (1 − food cost rate − operating cost rate) − orders × delivery cost per order
2. **Calculate, don't predict** (`app/utils/profit_engine.py`). Because profit is a formula, every scenario is an exact recalculation, not a model guess.
3. **Six pages**, all with cuisine, segment and subregion filters:
   - **Home:** portfolio KPIs, profit by channel, top 10 restaurants
   - **Overview:** profit by cuisine, segment, channel and subregion
   - **Profit Calculator:** the formulas, a live check against the data, channel economics, and a single-restaurant calculator
   - **What-If Simulator:** change commission, costs, order value or volume, and see how much commission the selection can absorb before profit hits $0
   - **Channel Shift:** move a share of Uber Eats or DoorDash orders to in-store or self-delivery and see the extra profit
   - **Sensitivity:** exact profit change for a small step in each driver, plus margin by segment

## How well it works

- **The numbers are exact.** The formula matches the dataset to within half a cent on every row, so there is no model error.
- **Why there's no machine-learning model.** My first version trained Random Forest and Gradient Boosting models to predict profit, and they scored R² above 0.95. That score was meaningless. Profit is built from the same columns the models were given, so they were just re-learning the formula. This is called **data leakage**. I replaced them with the exact formula.
- **Why the old optimiser was removed.** It stopped after one step and returned its starting guess, so its "45.4% uplift" was not a real result. The Channel Shift page replaces it with a calculation that shows its assumptions.

## Limitations

- **Sample data.** The dataset was provided for the internship. I could not confirm it comes from SkyCity itself, and because every profit figure follows a formula, it behaves like modelled data rather than real accounts.
- **Channel share columns don't add up.** For each restaurant, the four share columns sum to 103%–155%. The app ignores them and uses actual order counts.
- **No customer behaviour.** The data is a single month. Scenarios assume order volumes don't change when commission or prices change.
- **Channel Shift is an upper bound.** It assumes every moved customer keeps ordering at the same order value.
- **Same cost rates on every channel.** The data applies one food cost rate and one operating cost rate to all channels. Real in-store dining may cost more to serve.

## Run it locally

```bash
git clone https://github.com/Vipsa2207/SkyCity-Auckland-restaurants-bars-profit-dashboard.git
cd SkyCity-Auckland-restaurants-bars-profit-dashboard
pip install -r requirements.txt
streamlit run app/main.py
```

## Project files

| File | What it is |
|---|---|
| `app/main.py` | Home page |
| `app/pages/` | Overview, Profit Calculator, What-If Simulator, Channel Shift, Sensitivity |
| `app/utils/profit_engine.py` | Exact profit formulas, scenarios, break-even and sensitivity |
| `app/utils/data_loader.py` | Loads data and builds totals |
| `app/utils/plot_utils.py` | Shared styling and chart helpers |
| `data/` | Restaurant-level dataset (1,696 rows) |
| `docs/` | Revised research paper, executive summary, dashboard screenshot |

## Research paper

I authored a research paper on this project (JETIR paper ID 583938, accepted, not published).

- [Research paper (revised, November 2026)](docs/Research_Paper_Revised.pdf). It corrects the version submitted to JETIR: the 45.4% uplift and the R² ≈ 0.96–0.97 model scores are withdrawn, and the channel margins and order shares are corrected.
- [Executive summary (revised, November 2026)](docs/Executive_Summary_Revised.pdf)

## How I built it

I used AI to help write the Python code. I framed the business question, checked the outputs against the data, and wrote the findings. An AI-assisted review found that profit was a fixed formula, and I rebuilt the model pages around it.

## Author

**Vipsa Patel** — Data Analyst Intern, Unified Mentor · [LinkedIn](https://www.linkedin.com/in/vipsapatel)
