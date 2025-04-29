import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose

# Ensure the results directory exists
os.makedirs("results", exist_ok=True)

def detect_trend(df):
    """
    Fit a linear model Y = a + bX to detect overall trend.
    Returns the slope b (positive: up, negative: down).
    """
    ts = df.set_index('transaction_date').sort_index()['sales']
    ts = ts.reset_index()
    ts['time_idx'] = range(len(ts))
    X = ts[['time_idx']]
    y = ts['sales']
    lr = LinearRegression().fit(X, y)
    slope = lr.coef_[0]
    print(f"Linear trend slope: {slope:.2f} ({'upward' if slope>0 else 'downward' if slope<0 else 'flat'})")
    return slope

def decompose_series(df, period):
    """
    Decompose the time series into trend, seasonal, and residual.
    """
    ts = df.set_index('transaction_date').sort_index()['sales']
    result = seasonal_decompose(ts, model='additive', period=period, two_sided=False)
    plt.figure(figsize=(12, 8))
    result.plot()
    plt.suptitle("Time Series Decomposition", fontsize=16)
    plt.tight_layout()
    plt.savefig("results/decomposition_plot.png")
    plt.close()

def run_sales_trends_analysis(cleaned_transaction_df):
    """
    Extended sales trends analysis with:
      - Linear trend detection
      - Time series decomposition
      - Basic weekly/monthly/yearly plots
    """
    print("Running extended sales trends analysis...")

    # Convert to Pandas
    df = cleaned_transaction_df.toPandas()
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    df['sales'] = df['quantity'] * df['unit_price']

    # 1) Detect overall trend
    detect_trend(df)

    # 2) Decompose series (use weekly seasonality => period=52 if yearly, or 12 for monthly)
    decompose_series(df, period=52)

    # 3) Usual weekly/monthly/yearly trend plots
    df['week'] = df['transaction_date'].dt.to_period('W').apply(lambda r: r.start_time)
    df['month'] = df['transaction_date'].dt.to_period('M').apply(lambda r: r.start_time)
    df['year'] = df['transaction_date'].dt.year

    for label, grp in [('weekly', 'week'), ('monthly', 'month'), ('yearly', 'year')]:
        agg = df.groupby(grp)['sales'].sum().reset_index()
        plt.figure(figsize=(10, 6))
        if label == 'yearly':
            sns.barplot(data=agg, x=grp, y='sales', palette='muted')
        else:
            sns.lineplot(data=agg, x=grp, y='sales', marker='o')
            plt.xticks(rotation=45)
        plt.title(f"{label.capitalize()} Sales Trend")
        plt.tight_layout()
        plt.savefig(f"results/{label}_sales_trend.png")
        plt.close()

    print("Extended sales trends analysis completed.\n")
