import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure results directory exists
if not os.path.exists("results"):
    os.makedirs("results")

# Run Channel Comparison Analysis
def run_channel_comparison(cleaned_transaction_df):
    print("Running In-Store vs. Online Purchase Comparison...")

    # Convert Spark DataFrame to Pandas
    trans_df = cleaned_transaction_df.toPandas()

    # Compute sales (quantity * unit price)
    trans_df['sales'] = trans_df['quantity'] * trans_df['unit_price']

    # Convert transaction_date to datetime if not already
    trans_df['transaction_date'] = pd.to_datetime(trans_df['transaction_date'])

    # Aggregating total sales per channel
    channel_sales = trans_df.groupby('channel')['sales'].sum().reset_index()
    channel_sales.columns = ['Channel', 'Total Sales']

    # Average basket size per channel
    channel_avg_basket = trans_df.groupby('channel').agg(
        avg_basket_size=('sales', 'sum'),
        total_transactions=('transaction_date', 'count')
    ).reset_index()

    channel_avg_basket['avg_basket_size'] = channel_avg_basket['avg_basket_size'] / channel_avg_basket['total_transactions']

    # Customer count per channel
    customer_count = trans_df.groupby('channel')['customer_id'].nunique().reset_index()
    customer_count.columns = ['Channel', 'Customer Count']

    # Monthly sales trend per channel
    trans_df['month'] = trans_df['transaction_date'].dt.to_period('M')
    monthly_sales = trans_df.groupby(['channel', 'month']).agg(monthly_sales=('sales', 'sum')).reset_index()

    # Visualizations
    # 1. Bar Chart: Total sales by channel
    plt.figure(figsize=(8, 6))
    sns.barplot(data=channel_sales, x='Channel', y='Total Sales', palette='viridis')
    plt.title('Total Sales by Channel')
    plt.xlabel('Channel')
    plt.ylabel('Total Sales')
    plt.tight_layout()
    plt.savefig("results/total_sales_by_channel.png")
    plt.close()

    # 2. Line Chart: Monthly sales trend by channel
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=monthly_sales, x='month', y='monthly_sales', hue='channel', marker='o')
    plt.title('Monthly Sales Trend by Channel')
    plt.xlabel('Month')
    plt.ylabel('Monthly Sales')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("results/monthly_sales_trend_by_channel.png")
    plt.close()

    # 3. Bar Chart: Avg basket size per channel
    plt.figure(figsize=(8, 6))
    sns.barplot(data=channel_avg_basket, x='channel', y='avg_basket_size', palette='viridis')
    plt.title('Average Basket Size by Channel')
    plt.xlabel('Channel')
    plt.ylabel('Average Basket Size')
    plt.tight_layout()
    plt.savefig("results/avg_basket_size_by_channel.png")
    plt.close()

    # 4. Bar Chart: Customer count by channel
    plt.figure(figsize=(8, 6))
    sns.barplot(data=customer_count, x='Channel', y='Customer Count', palette='viridis')
    plt.title('Customer Count by Channel')
    plt.xlabel('Channel')
    plt.ylabel('Customer Count')
    plt.tight_layout()
    plt.savefig("results/customer_count_by_channel.png")
    plt.close()

    # Save summary metrics as CSV
    summary_metrics = pd.merge(channel_sales, channel_avg_basket[['channel', 'avg_basket_size']], on='channel')
    summary_metrics = pd.merge(summary_metrics, customer_count, on='Channel')
    summary_metrics.to_csv("results/channel_comparison_summary.csv", index=False)

    print("In-Store vs. Online Purchase Comparison completed.\nInsights and charts saved.")

