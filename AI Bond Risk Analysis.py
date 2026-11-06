import pandas as pd

portfolio = pd.read_csv("bond_portfolio_data.csv")
yield_curve = pd.read_csv("yield_curve_history.csv")
scenarios = pd.read_csv("monte_carlo_scenarios.csv")
portfolio.head()
portfolio.info()

portfolio["MarketValue_INR"].sum()

portfolio_duration = (
portfolio["ModifiedDuration"] *
portfolio["MarketValue_INR"]
).sum() / portfolio["MarketValue_INR"].sum()

portfolio_convexity = (
portfolio["Convexity"] *
portfolio["MarketValue_INR"]
).sum() / portfolio["MarketValue_INR"].sum()

portfolio["DV01"] = (
portfolio["ModifiedDuration"]
*
portfolio["DirtyPrice"]
*
0.0001
)

import matplotlib.pyplot as plt

latest_curve = yield_curve[
    yield_curve["CurveDate"] ==
    yield_curve["CurveDate"].max()
]

plt.figure(figsize=(10,5))

plt.plot(
    latest_curve["Tenor_Years"],
    latest_curve["Yield"],
    marker='o'
)

plt.title("Yield Curve")
plt.xlabel("Tenor (Years)")
plt.ylabel("Yield (%)")
plt.grid(True)

plt.show()

print(portfolio.columns.tolist())
portfolio.head()

total_market_value = portfolio["MarketValue_INR"].sum()

print(f"Portfolio Market Value: ₹{total_market_value:,.2f}")

portfolio_duration = (
    portfolio["ModifiedDuration"]
    * portfolio["MarketValue_INR"]
).sum() / portfolio["MarketValue_INR"].sum()

print("Portfolio Duration:", round(portfolio_duration,2))

portfolio_convexity = (
    portfolio["Convexity"]
    * portfolio["MarketValue_INR"]
).sum() / portfolio["MarketValue_INR"].sum()

print("Portfolio Convexity:", round(portfolio_convexity,2))

portfolio["DV01_INR"] = (
    portfolio["ModifiedDuration"]
    *
    portfolio["DirtyPrice"]
    *
    portfolio["Quantity"]
    *
    0.0001
)

total_dv01 = portfolio["DV01_INR"].sum()

print(f"Portfolio DV01: ₹{total_dv01:,.2f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))

plt.hist(
    portfolio["ModifiedDuration"],
    bins=10
)

plt.title("Duration Distribution")
plt.xlabel("Modified Duration")
plt.ylabel("Number of Bonds")

plt.show()

rating_risk = (
    portfolio.groupby("CreditRating")
    ["MarketValue_INR"]
    .sum()
    .sort_values(ascending=False)
)

print(rating_risk)

rating_risk.plot(
    kind="bar"
)

plt.title("Market Value by Credit Rating")
plt.show()

dy = 0.005

portfolio["Shock50bp"] = (
    -portfolio["ModifiedDuration"]*dy
    +
    0.5*
    portfolio["Convexity"]*
    (dy**2)
)

dy = 0.01

portfolio["Shock100bp"] = (
    -portfolio["ModifiedDuration"]*dy
    +
    0.5*
    portfolio["Convexity"]*
    (dy**2)
)

dy = 0.02

portfolio["Shock200bp"] = (
    -portfolio["ModifiedDuration"]*dy
    +
    0.5*
    portfolio["Convexity"]*
    (dy**2)
)

stress_loss_100bp = (
    portfolio["Shock100bp"]
    *
    portfolio["MarketValue_INR"]
).sum()

print(
    f"Portfolio Loss (+100bps): ₹{stress_loss_100bp:,.2f}"
)

yield_curve["Yield_Shock100bp"] = (
    yield_curve["Yield"] + 0.01
)
plt.figure(figsize=(10,5))

latest_curve = yield_curve[
    yield_curve["CurveDate"] ==
    yield_curve["CurveDate"].max()
]

plt.plot(
    latest_curve["Tenor_Years"],
    latest_curve["Yield"],
    marker='o',
    label='Original'
)

plt.plot(
    latest_curve["Tenor_Years"],
    latest_curve["Yield"]+0.01,
    marker='s',
    label='100bp Shock'
)

plt.legend()
plt.grid(True)
plt.show()

portfolio_duration = (
    portfolio["ModifiedDuration"] *
    portfolio["MarketValue_INR"]
).sum() / portfolio["MarketValue_INR"].sum()

print("Portfolio Duration:", round(portfolio_duration, 2))

portfolio_convexity = (
    portfolio["Convexity"] *
    portfolio["MarketValue_INR"]
).sum() / portfolio["MarketValue_INR"].sum()

print("Portfolio Convexity:", round(portfolio_convexity, 2))

portfolio["DV01_INR"] = (
    portfolio["ModifiedDuration"]
    * portfolio["DirtyPrice"]
    * portfolio["Quantity"]
    * 0.0001
)

total_dv01 = portfolio["DV01_INR"].sum()

print(f"Portfolio DV01: ₹{total_dv01:,.2f}")

dy = 0.01

portfolio["Shock100bp"] = (
    -portfolio["ModifiedDuration"] * dy
    +
    0.5 * portfolio["Convexity"] * (dy**2)
)

stress_loss_100bp = (
    portfolio["Shock100bp"]
    * portfolio["MarketValue_INR"]
).sum()

print(f"Portfolio Loss (+100bps): ₹{stress_loss_100bp:,.2f}")

import numpy as np

n_scenarios = 10000

rate_shocks = np.random.normal(
    loc=0,
    scale=0.01,
    size=n_scenarios
)
portfolio_pnl = (
    -portfolio_duration * rate_shocks
    +
    0.5 * portfolio_convexity * (rate_shocks**2)
)

portfolio_pnl = (
    portfolio_pnl
    * portfolio["MarketValue_INR"].sum()
)

var95 = np.percentile(portfolio_pnl, 5)

print(f"95% VaR: ₹{var95:,.2f}")

es95 = portfolio_pnl[
    portfolio_pnl <= var95
].mean()

print(f"Expected Shortfall: ₹{es95:,.2f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))

plt.hist(
    portfolio_pnl,
    bins=50
)

plt.axvline(
    var95,
    linestyle="--",
    label=f"VaR 95% = {var95:,.0f}"
)

plt.title("Monte Carlo Portfolio P&L Distribution")
plt.xlabel("Portfolio P&L (INR)")
plt.ylabel("Frequency")
plt.legend()

plt.show()

portfolio["RiskScore"] = (
    portfolio["ModifiedDuration"] * 0.4
    +
    portfolio["Convexity"] * 0.3
    +
    portfolio["YieldToMaturity"] * 100 * 0.2
    +
    portfolio["OAS_bps"] * 0.1
)


portfolio[
    ["BondID","RiskScore"]
].head()

features = [
    "ModifiedDuration",
    "Convexity",
    "YieldToMaturity",
    "OAS_bps",
    "ZSpread_bps",
    "SpreadOverBenchmark_bps",
    "PortfolioWeight"
]

from sklearn.model_selection import train_test_split

X = portfolio[features]
y = portfolio["RiskScore"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)

rf_predictions = rf.predict(X_test)

from sklearn.metrics import r2_score

print(
    "Random Forest R²:",
    round(
        r2_score(
            y_test,
            rf_predictions
        ),
        4
    )
)

import pandas as pd

importance = pd.DataFrame({
    "Feature": features,
    "Importance": rf.feature_importances_
})

importance.sort_values(
    by="Importance",
    ascending=False
)

import matplotlib.pyplot as plt

importance_sorted = importance.sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(10,5))

plt.barh(
    importance_sorted["Feature"],
    importance_sorted["Importance"]
)

plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")

plt.show()

from xgboost import XGBRegressor
from sklearn.metrics import r2_score

xgb = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    random_state=42
)

xgb.fit(X_train, y_train)

xgb_predictions = xgb.predict(X_test)

xgb_r2 = r2_score(
    y_test,
    xgb_predictions
)

print("XGBoost R²:", round(xgb_r2,4))

comparison = pd.DataFrame({
    "Model":[
        "Random Forest",
        "XGBoost"
    ],
    "R2":[
        r2_score(y_test, rf_predictions),
        r2_score(y_test, xgb_predictions)
    ]
})

comparison

def bond_risk_agent(
    duration,
    convexity,
    var95
):

    if abs(var95) > 2000000:
        return "HIGH RISK"

    elif duration > 5:
        return "MODERATE RISK"

    else:
        return "LOW RISK"
risk_status = bond_risk_agent(
    portfolio_duration,
    portfolio_convexity,
    var95
)

print("Portfolio Risk Status:", risk_status)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

nn = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(8, activation="relu"),
    Dense(1)
])

nn.compile(
    optimizer='adam',
    loss='mse'
)

nn.fit(
    X_train_scaled,
    y_train,
    epochs=50,
    batch_size=16,
    verbose=0
)

nn_predictions = nn.predict(X_test_scaled)

nn_r2 = r2_score(
    y_test,
    nn_predictions
)

print("Neural Network R²:", round(nn_r2,4))

portfolio_risk_score = (
    abs(var95)/100000
    +
    portfolio_duration*5
    +
    portfolio_convexity*0.5
)

print("Portfolio Risk Score:",
      round(portfolio_risk_score,2))

if portfolio_risk_score < 50:
    risk_band = "LOW"

elif portfolio_risk_score < 75:
    risk_band = "MODERATE"

else:
    risk_band = "HIGH"

print(risk_band)

from tensorflow.keras import Input

nn = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1)
])

portfolio.to_csv(
    "bond_risk_results.csv",
    index=False
)

print(type(portfolio_pnl))
print(len(portfolio_pnl))
print(portfolio_pnl[:5])

mc_results = pd.DataFrame({
    "Scenario": range(len(portfolio_pnl)),
    "PnL": portfolio_pnl
})

mc_results.head()

mc_results.to_csv("monte_carlo_results.csv", index=False)

print("Monte Carlo results exported successfully!")

summary = pd.DataFrame({
    "Metric":[
        "Portfolio Market Value",
        "Average Duration",
        "Average Convexity",
        "VaR 95%",
        "Expected Shortfall",
        "Portfolio Risk Score"
    ],
    "Value":[
        portfolio["MarketValue_INR"].sum(),
        portfolio_duration,
        portfolio_convexity,
        var95,
        es95,
        portfolio_risk_score
    ]
})

summary.to_csv("portfolio_summary.csv",index=False)

comparison.to_csv(
    "model_comparison.csv",
    index=False
)

portfolio.to_csv(
    "portfolio_details.csv",
    index=False
)


  
