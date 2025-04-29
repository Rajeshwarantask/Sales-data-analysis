import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_seasonality(monthly_sales_df):
    print("\n[Seasonality Analysis Started]")

    # Ensure the results directory exists
    if not os.path.exists("results"):
        os.makedirs("results")

    # Add 'month' and 'year' features
    monthly_sales_df["month_num"] = monthly_sales_df["ds"].dt.month
    monthly_sales_df["year"] = monthly_sales_df["ds"].dt.year

    # Aggregate total sales by month number across all years
    seasonal_pattern = monthly_sales_df.groupby("month_num").agg({
        "y": ["sum", "mean", "count"]
    }).reset_index()
    seasonal_pattern.columns = ["month", "total_sales", "average_sales", "transaction_count"]

    # Save seasonal insights
    seasonal_pattern.to_csv("results/seasonality_summary.csv", index=False)

    # Plot seasonal total sales
    plt.figure(figsize=(10, 6))
    sns.barplot(data=seasonal_pattern, x="month", y="total_sales", palette="Blues_d")
    plt.title("Total Sales by Month")
    plt.xlabel("Month (1=Jan, 12=Dec)")
    plt.ylabel("Total Sales")
    plt.tight_layout()
    plt.savefig("results/total_sales_by_month.png")
    plt.close()

    # Plot seasonal average sales
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=seasonal_pattern, x="month", y="average_sales", marker="o", color="green")
    plt.title("Average Sales by Month")
    plt.xlabel("Month (1=Jan, 12=Dec)")
    plt.ylabel("Average Sales")
    plt.tight_layout()
    plt.savefig("results/average_sales_by_month.png")
    plt.close()

    print("[Seasonality Analysis Completed] — Files saved to 'results/'\n")
