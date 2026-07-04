import pandas as pd
import numpy as np
import os

# ── Path to CSV ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "SkyCity Auckland Restaurants & Bars.csv")

# ── Load & Cache Data ─────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    """Load raw CSV and return a cleaned DataFrame."""
    df = pd.read_csv(DATA_PATH)
    df = _clean(df)
    df = _engineer_features(df)
    return df


# ── Cleaning ──────────────────────────────────────────────────────────────────
def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates, fix types, strip whitespace."""
    df = df.drop_duplicates()
    df.columns = df.columns.str.strip()

    # Ensure numeric columns are correct type
    numeric_cols = [
        "GrowthFactor", "AOV", "MonthlyOrders",
        "InStoreOrders", "UberEatsOrders", "DoorDashOrders", "SelfDeliveryOrders",
        "InStoreRevenue", "UberEatsRevenue", "DoorDashRevenue", "SelfDeliveryRevenue",
        "COGSRate", "OPEXRate", "CommissionRate",
        "DeliveryRadiusKM", "DeliveryCostPerOrder", "SD_DeliveryTotalCost",
        "InStoreNetProfit", "UberEatsNetProfit", "DoorDashNetProfit", "SelfDeliveryNetProfit",
        "InStoreShare", "UE_share", "DD_share", "SD_share"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill any remaining NaNs in numeric cols with median
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    return df


# ── Feature Engineering ───────────────────────────────────────────────────────
def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create model-ready and dashboard-ready features."""

    # Total Revenue & Net Profit
    df["TotalRevenue"] = (
        df["InStoreRevenue"] + df["UberEatsRevenue"] +
        df["DoorDashRevenue"] + df["SelfDeliveryRevenue"]
    )
    df["TotalNetProfit"] = (
        df["InStoreNetProfit"] + df["UberEatsNetProfit"] +
        df["DoorDashNetProfit"] + df["SelfDeliveryNetProfit"]
    )

    # Profit per Order
    df["ProfitPerOrder"] = df["TotalNetProfit"] / df["MonthlyOrders"].replace(0, np.nan)

    # Profit Margin %
    df["ProfitMargin"] = (df["TotalNetProfit"] / df["TotalRevenue"].replace(0, np.nan)) * 100

    # Cost to Revenue Ratios
    df["COGS_Revenue_Ratio"]  = df["COGSRate"]
    df["OPEX_Revenue_Ratio"]  = df["OPEXRate"]
    df["TotalCostRate"]       = df["COGSRate"] + df["OPEXRate"]

    # Interaction Terms (for ML models)
    df["Commission_UE"]   = df["CommissionRate"] * df["UE_share"]
    df["Commission_DD"]   = df["CommissionRate"] * df["DD_share"]
    df["DeliveryCost_SD"] = df["DeliveryCostPerOrder"] * df["SD_share"]

    # Growth-Adjusted Revenue
    df["GrowthAdjustedRevenue"] = df["TotalRevenue"] * df["GrowthFactor"]

    # Channel Efficiency (profit per share)
    df["InStore_Efficiency"]  = df["InStoreNetProfit"]  / df["InStoreShare"].replace(0, np.nan)
    df["UE_Efficiency"]       = df["UberEatsNetProfit"] / df["UE_share"].replace(0, np.nan)
    df["DD_Efficiency"]       = df["DoorDashNetProfit"] / df["DD_share"].replace(0, np.nan)
    df["SD_Efficiency"]       = df["SelfDeliveryNetProfit"] / df["SD_share"].replace(0, np.nan)

    # Delivery vs InStore Revenue Split
    df["DeliveryRevenue"] = (
        df["UberEatsRevenue"] + df["DoorDashRevenue"] + df["SelfDeliveryRevenue"]
    )
    df["DeliveryShare"] = df["DeliveryRevenue"] / df["TotalRevenue"].replace(0, np.nan)

    return df


# ── Filter Helpers ────────────────────────────────────────────────────────────
def get_cuisines(df: pd.DataFrame) -> list:
    return sorted(df["CuisineType"].dropna().unique().tolist())

def get_segments(df: pd.DataFrame) -> list:
    return sorted(df["Segment"].dropna().unique().tolist())

def get_subregions(df: pd.DataFrame) -> list:
    return sorted(df["Subregion"].dropna().unique().tolist())

def filter_data(df, cuisines=None, segments=None, subregions=None) -> pd.DataFrame:
    """Apply sidebar filters and return filtered DataFrame."""
    if cuisines:
        df = df[df["CuisineType"].isin(cuisines)]
    if segments:
        df = df[df["Segment"].isin(segments)]
    if subregions:
        df = df[df["Subregion"].isin(subregions)]
    return df