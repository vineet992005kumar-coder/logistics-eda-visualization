"""
Week 3 Task - Advanced Data Analysis and Visualization in Logistics
Yuva Intern

Creates a hypothetical logistics dataset, runs exploratory data analysis (EDA)
and produces 7 charts with matplotlib and seaborn.
Charts are saved in the "charts" folder.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

rng = np.random.default_rng(2025)
os.makedirs("charts", exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=1.0)


# ------------------------------------------------------------
# 1. Data simulation
# ------------------------------------------------------------
def make_dataset(n=6000):
    date = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    hub = rng.choice(["Hub A", "Hub B", "Hub C"], n, p=[0.40, 0.35, 0.25])
    carrier = rng.choice(["FastShip", "QuickMove", "CityCourier"], n, p=[0.40, 0.35, 0.25])
    mode = rng.choice(["Road", "Rail", "Air"], n, p=[0.65, 0.20, 0.15])

    distance = np.where(mode == "Road", rng.gamma(2.5, 120, n),
               np.where(mode == "Rail", rng.gamma(3.0, 250, n), rng.gamma(3.0, 450, n)))
    distance = np.clip(distance, 20, 3500)
    weight = rng.gamma(2.2, 14, n).clip(1, 250)
    volume = (weight * rng.uniform(0.8, 1.6, n)).round(1)

    rate = {"Road": 0.9, "Rail": 0.55, "Air": 3.4}
    cost = (np.array([rate[m] for m in mode]) * distance + 1.2 * weight + 25
            + rng.normal(0, 40, n)).clip(15)

    speed = {"Road": 600, "Rail": 450, "Air": 1500}          # km per day
    days = distance / np.array([speed[m] for m in mode]) + 1 + rng.gamma(1.5, 0.6, n)

    month = pd.DatetimeIndex(date).month
    peak = np.isin(month, [10, 11, 12])
    days = days + peak * rng.gamma(2.0, 0.5, n)               # peak season congestion
    days = days + (carrier == "CityCourier") * rng.gamma(2.0, 0.45, n)
    days = days + (hub == "Hub C") * rng.gamma(1.5, 0.35, n)

    promised = 4.5 + (distance > 1000) * 2
    df = pd.DataFrame({
        "shipment_id": np.arange(1, n + 1), "ship_date": date, "hub": hub, "carrier": carrier,
        "mode": mode, "distance_km": distance.round(1), "weight_kg": weight.round(1),
        "volume_units": volume, "transport_cost": cost.round(2), "delivery_days": days.round(2),
        "promised_days": promised,
    })
    df["delayed"] = (df["delivery_days"] > df["promised_days"]).astype(int)
    df["month"] = df["ship_date"].dt.month
    df["weekday"] = df["ship_date"].dt.day_name()
    df["cost_per_km"] = df["transport_cost"] / df["distance_km"]
    return df


df = make_dataset()
df.to_csv("logistics_dataset.csv", index=False)

# ------------------------------------------------------------
# 2. Exploratory analysis: central tendency, spread, correlation
# ------------------------------------------------------------
num = ["distance_km", "weight_kg", "volume_units", "transport_cost", "delivery_days"]
print("Shape:", df.shape)
print("\nSummary statistics:")
print(df[num].describe().round(2))
print("\nMedian:\n", df[num].median().round(2))
print("\nSkewness:\n", df[num].skew().round(2))
print("\nOverall delay rate: %.1f%%" % (100 * df["delayed"].mean()))
print("\nCorrelation matrix:")
print(df[num].corr().round(2))


# ------------------------------------------------------------
# 3. Visualizations
# ------------------------------------------------------------
def save(name):
    plt.tight_layout()
    plt.savefig(f"charts/{name}.png", dpi=150)
    plt.close()


# Chart 1 - distribution of delivery time (histogram + KDE)
plt.figure(figsize=(8, 4.5))
sns.histplot(df["delivery_days"], bins=40, kde=True, color="#2E74B5")
plt.axvline(df["delivery_days"].mean(), color="red", ls="--", label=f"Mean {df['delivery_days'].mean():.1f} days")
plt.axvline(df["delivery_days"].median(), color="green", ls="--", label=f"Median {df['delivery_days'].median():.1f} days")
plt.title("Distribution of Delivery Time"); plt.xlabel("Delivery time (days)"); plt.ylabel("Shipments")
plt.legend(); save("01_delivery_time_distribution")

# Chart 2 - monthly shipment volume and delay rate (trend)
monthly = df.groupby("month").agg(shipments=("shipment_id", "count"), delay_rate=("delayed", "mean"))
fig, ax1 = plt.subplots(figsize=(8, 4.5))
ax1.bar(monthly.index, monthly["shipments"], color="#9DC3E6")
ax1.set_xlabel("Month"); ax1.set_ylabel("Shipments"); ax1.set_xticks(range(1, 13)); ax1.grid(axis="x", visible=False)
ax2 = ax1.twinx()
ax2.plot(monthly.index, monthly["delay_rate"] * 100, color="#C00000", marker="o")
ax2.set_ylabel("Delay rate (%)", color="#C00000"); ax2.grid(False)
plt.title("Monthly Shipment Volume and Delay Rate"); save("02_monthly_trend")

# Chart 3 - transport cost by mode (boxplot)
plt.figure(figsize=(8, 4.5))
sns.boxplot(data=df, x="mode", y="transport_cost", order=["Road", "Rail", "Air"],
            palette=["#9DC3E6", "#A9D18E", "#F4B183"], hue="mode", legend=False)
plt.title("Transport Cost by Mode"); plt.xlabel("Transport mode"); plt.ylabel("Transport cost"); save("03_cost_by_mode")

# Chart 4 - distance vs cost (scatter)
plt.figure(figsize=(8, 4.5))
sns.scatterplot(data=df.sample(1500, random_state=1), x="distance_km", y="transport_cost",
                hue="mode", alpha=0.6, s=25)
plt.title("Distance vs Transport Cost"); plt.xlabel("Distance (km)"); plt.ylabel("Transport cost"); save("04_distance_vs_cost")

# Chart 5 - correlation heatmap
plt.figure(figsize=(7, 5.5))
sns.heatmap(df[num].corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Between Key Variables"); save("05_correlation_heatmap")

# Chart 6 - delay rate by carrier and hub (bar)
plt.figure(figsize=(8, 4.5))
sns.barplot(data=df, x="carrier", y="delayed", hue="hub", errorbar=None)
plt.ylabel("Delay rate"); plt.xlabel("Carrier"); plt.title("Delay Rate by Carrier and Hub")
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}")); save("06_delay_by_carrier_hub")

# Chart 7 - average cost per km by mode and hub (heatmap)
pivot = df.pivot_table(index="hub", columns="mode", values="cost_per_km", aggfunc="mean")[["Road", "Rail", "Air"]]
plt.figure(figsize=(6.5, 4))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd")
plt.title("Average Cost per km by Hub and Mode"); save("07_cost_per_km_heatmap")

print("\nSaved 7 charts to the charts folder.")
