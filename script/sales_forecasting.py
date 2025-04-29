import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split
import joblib

# Ensure the results directory exists
if not os.path.exists("results"):
    os.makedirs("results")

# Evaluate regression model
def evaluate_regression_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2_score(y_true, y_pred)
    }

# Hyperparameter tuning
def hyperparameter_tuning(model, X_train, y_train):
    if isinstance(model, RandomForestRegressor):
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
    elif isinstance(model, XGBRegressor):
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }
    else:
        return model

    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best Parameters for {model.__class__.__name__}: {grid_search.best_params_}")
    return grid_search.best_estimator_

# Run forecasting process
def run_forecasting(cleaned_transaction_df):
    print("Preparing sales forecasting...")

    trans_df = cleaned_transaction_df.toPandas()
    trans_df['transaction_date'] = pd.to_datetime(trans_df['transaction_date'])
    trans_df['sales'] = trans_df['quantity'] * trans_df['unit_price']
    daily_sales = trans_df.groupby('transaction_date')['sales'].sum().reset_index()
    daily_sales.columns = ['ds', 'y']

    # Prophet Forecasting
    prophet_model = Prophet()
    prophet_model.fit(daily_sales)

    future = prophet_model.make_future_dataframe(periods=30)
    forecast = prophet_model.predict(future)

    # Prophet Plot
    fig1 = prophet_model.plot(forecast)
    plt.title("Sales Forecast (Prophet)")
    plt.tight_layout()
    fig1.savefig("results/sales_forecast_prophet.png")
    plt.close()

    # Prophet Seasonality Plot
    fig2 = prophet_model.plot_components(forecast)
    plt.tight_layout()
    fig2.savefig("results/sales_forecast_seasonality.png")
    plt.close()

    # Add time index
    df_ml = daily_sales.copy()
    df_ml['day'] = range(len(df_ml))
    X = df_ml[['day']]
    y = df_ml['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(),
        "XGBoost": XGBRegressor()
    }

    results = []

    for name, model in models.items():
        print(f"Training {name}...")
        model = hyperparameter_tuning(model, X_train, y_train)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate_regression_model(y_test, y_pred)
        metrics["Model"] = name
        results.append(metrics)

        # Actual vs Predicted Plot
        plt.figure(figsize=(8, 5))
        plt.plot(y_test.values, label='Actual')
        plt.plot(y_pred, label='Predicted')
        plt.title(f"{name} - Sales Forecast")
        plt.xlabel("Days")
        plt.ylabel("Sales")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/sales_forecast_{name.lower().replace(' ', '_')}.png")
        plt.close()

    results_df = pd.DataFrame(results).set_index("Model")
    print("\nModel Comparison (Sales Forecasting):\n", results_df)
    results_df.to_csv("results/sales_forecast_comparison.csv")

    # Column chart for comparison
    results_df[['MAE', 'RMSE']].plot(kind='bar', figsize=(10, 6), title='Model Error Comparison')
    plt.ylabel("Error")
    plt.tight_layout()
    plt.savefig("results/model_error_comparison.png")
    plt.close()

    best_model_name = results_df["R2 Score"].idxmax()
    best_model = [model for name, model in models.items() if name == best_model_name][0]
    joblib.dump(best_model, "results/best_sales_forecast_model.pkl")

    # Weekly/Monthly/Yearly trend analysis
    sales_trend_df = trans_df.copy()
    sales_trend_df['week'] = sales_trend_df['transaction_date'].dt.to_period('W').apply(lambda r: r.start_time)
    sales_trend_df['month'] = sales_trend_df['transaction_date'].dt.to_period('M').apply(lambda r: r.start_time)
    sales_trend_df['year'] = sales_trend_df['transaction_date'].dt.year

    weekly_sales = sales_trend_df.groupby('week')['sales'].sum().reset_index()
    monthly_sales = sales_trend_df.groupby('month')['sales'].sum().reset_index()
    yearly_sales = sales_trend_df.groupby('year')['sales'].sum().reset_index()

    weekly_sales.to_csv("results/weekly_sales_trend.csv", index=False)
    monthly_sales.to_csv("results/monthly_sales_trend.csv", index=False)
    yearly_sales.to_csv("results/yearly_sales_trend.csv", index=False)

    # Plot trends
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=weekly_sales, x='week', y='sales')
    plt.title("Weekly Sales Trend")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("results/weekly_sales_trend.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=monthly_sales, x='month', y='sales')
    plt.title("Monthly Sales Trend")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("results/monthly_sales_trend.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.barplot(data=yearly_sales, x='year', y='sales')
    plt.title("Yearly Sales Trend")
    plt.tight_layout()
    plt.savefig("results/yearly_sales_trend.png")
    plt.close()

    # Heatmap: Monthly sales across years
    heatmap_data = sales_trend_df.copy()
    heatmap_data['month'] = heatmap_data['transaction_date'].dt.month
    heatmap_data['year'] = heatmap_data['transaction_date'].dt.year
    pivot_table = heatmap_data.pivot_table(values='sales', index='month', columns='year', aggfunc='sum')

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap="YlGnBu")
    plt.title("Monthly Sales Heatmap by Year")
    plt.tight_layout()
    plt.savefig("results/monthly_sales_heatmap.png")
    plt.close()

    # Top months and weeks
    top_months = monthly_sales.sort_values(by='sales', ascending=False).head(3)
    top_weeks = weekly_sales.sort_values(by='sales', ascending=False).head(3)
    print("\nTop 3 Sales Months:\n", top_months)
    print("\nTop 3 Sales Weeks:\n", top_weeks)

    print("Sales forecasting and analysis completed.\n")
