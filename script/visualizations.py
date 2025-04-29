import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Ensure directories exist
os.makedirs('visualizations', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Load your sales data
sales_data = pd.read_csv('C:/Users/Lenovo/OneDrive/Desktop/sales big data/data/retail_data.csv')
customer_df = pd.read_csv('C:/Users/Lenovo/OneDrive/Desktop/sales big data/data/retail_data.csv')

# Ensure date column is in datetime format
sales_data['transaction_date'] = pd.to_datetime(sales_data['transaction_date'])

# Function to plot sales trend over time
def plot_sales_trend():
    sales_trend = sales_data.resample('ME', on='transaction_date').sum()
    plt.figure(figsize=(12, 6))
    plt.plot(sales_trend.index, sales_trend['total_sales'], marker='o')
    plt.title('Sales Trend Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45)
    plt.grid()
    plt.tight_layout()
    plt.savefig('visualizations/sales_trend.png')
    plt.show()

# Function to plot customer spending vs transaction count
def plot_spending_vs_transactions():
    plt.figure(figsize=(50, 16))
    filtered_data = customer_df[
        (customer_df['transactions_count'] <= 100) & 
        (customer_df['average_spent'] <= 500)
    ]
    plt.scatter(
        filtered_data['transactions_count'], 
        filtered_data['average_spent'], 
        alpha=0.5, 
        color='blue'
    )
    plt.xlim(20, 100)
    plt.ylim(100, 600)
    plt.title('Customer Spending vs. Transactions Count')
    plt.xlabel('Transactions Count')
    plt.ylabel('Average Spending')
    plt.savefig('results/spending_vs_transactions.png')
    plt.close()

# Function to plot churn rate by age
def plot_churn_rate_by_age(df):
    sample_size = 100000
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    plt.figure(figsize=(20, 16))
    sns.countplot(data=df, x='age', hue='churn')
    plt.title('Churn Rate by Age')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.xticks(np.arange(18, 82, 2))
    plt.yticks(np.arange(0, 11000, 2000))
    plt.legend(title='Churn', loc='upper right', labels=['Not Churned', 'Churned'])
    plt.xticks(rotation=45)
    plt.grid()
    plt.tight_layout()
    plt.savefig('results/churn_rate_by_age.png')
    plt.close()

# Function to plot sales histogram distribution
def plot_sales_histogram():
    plt.figure(figsize=(26, 16))
    plt.hist(sales_data['total_sales'], bins=30, edgecolor='k', alpha=0.7)
    plt.title('Sales Amount Distribution')
    plt.xlabel('Sales Amount')
    plt.ylabel('Frequency')
    plt.grid()
    plt.savefig('results/sales_distribution.png')
    plt.close()

# Function to plot radar chart for sales by product category
def plot_radar_chart():
    categories = list(customer_df['product_category'].unique())
    values = [customer_df[customer_df['product_category'] == category]['total_sales'].sum() for category in categories]
    num_vars = len(categories)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='red', alpha=0.25)
    ax.plot(angles, values, color='red', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)

    plt.title('Sales by Product Category')
    plt.savefig('results/sales_by_product_category.png')
    plt.close()

# Product demand analysis with polynomial regression
def plot_product_demand_analysis():
    sales_data['month_year'] = sales_data['transaction_date'].dt.to_period('M')
    demand_trend = sales_data.groupby('month_year').agg({
        'total_sales': 'sum',
        'discount_applied': 'mean',
        'unit_price': 'mean',
    }).reset_index()

    demand_trend['month_year'] = demand_trend['month_year'].dt.to_timestamp()
    demand_trend['time_index'] = np.arange(len(demand_trend))
    
    X = demand_trend['time_index'].values
    y = demand_trend['total_sales'].values
    coefficients = np.polyfit(X, y, 3)
    regression_line = np.polyval(coefficients, X)

    plt.figure(figsize=(12, 6))
    plt.plot(demand_trend['month_year'], demand_trend['total_sales'], label='Actual Sales', marker='o', color='blue')
    plt.plot(demand_trend['month_year'], regression_line, label='Predicted Sales (Polynomial)', linestyle='--', color='red')
    plt.title('Product Demand Analysis with Polynomial Prediction')
    plt.xlabel('Date')
    plt.ylabel('Total Sales')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('results/product_demand_analysis_polynomial.png')
    plt.show()

# Confusion matrix display
def plot_confusion_matrix_custom(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig("results/confusion_matrix.png")
    plt.show()

# Correlation heatmap
def plot_correlation_heatmap(df):
    plt.figure(figsize=(12, 8))
    numeric_df = df.select_dtypes(include='number')
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('results/correlation_heatmap.png')
    plt.close()

# KDE + Histogram plot

def plot_sales_distribution(df):
    # Convert PySpark DataFrame to Pandas DataFrame for plotting
    df_pandas = df.toPandas()
    # Check the column names
    print(df_pandas.columns)
    # Choose the correct column name (check if it exists)
    column = 'total_sales'  # Replace with the actual column name
    if column in df_pandas.columns:
        # Plot the distribution of the sales column
        plt.figure(figsize=(10, 6))
        sns.histplot(df_pandas[column], kde=True, bins=30)
        plt.title(f'{column} Distribution')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()
    else:
        print(f"Column '{column}' not found in the DataFrame.")

# Boxplot of sales by category
def plot_sales_by_category(df, category_col='store_state', sales_col='total_sales'):
    df_pandas = df.toPandas()

    if category_col in df_pandas.columns and sales_col in df_pandas.columns:
        plt.figure(figsize=(12, 6))
        order = df_pandas.groupby(category_col)[sales_col].median().sort_values(ascending=False).index
        sns.boxplot(x=category_col, y=sales_col, data=df_pandas, order=order)
        plt.title(f'{category_col} vs {sales_col}')
        plt.xlabel(category_col)
        plt.ylabel(sales_col)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    else:
        print(f"One or both columns '{category_col}' and '{sales_col}' not found in the DataFrame.")


# Monthly and weekly trends
def plot_seasonal_sales_trends():
    sales_data['month'] = sales_data['transaction_date'].dt.month
    sales_data['week'] = sales_data['transaction_date'].dt.isocalendar().week

    monthly_sales = sales_data.groupby('month')['total_sales'].sum().reset_index()
    weekly_sales = sales_data.groupby('week')['total_sales'].sum().reset_index()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=monthly_sales, x='month', y='total_sales', marker='o')
    plt.title('Monthly Sales Trend')
    plt.xlabel('Month')
    plt.ylabel('Total Sales')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/monthly_sales_trend.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=weekly_sales, x='week', y='total_sales', marker='o')
    plt.title('Weekly Sales Trend')
    plt.xlabel('Week Number')
    plt.ylabel('Total Sales')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/weekly_sales_trend.png')
    plt.close()

# Product seasonality
def plot_product_seasonality(top_n=5):
    sales_data['month'] = sales_data['transaction_date'].dt.month
    top_products = sales_data.groupby('product_id')['total_sales'].sum().sort_values(ascending=False).head(top_n).index
    seasonal_df = sales_data[sales_data['product_id'].isin(top_products)]

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=seasonal_df, x='month', y='total_sales', hue='product_id', estimator='sum')
    plt.title(f'Seasonal Trend of Top {top_n} Products')
    plt.xlabel('Month')
    plt.ylabel('Total Sales')
    plt.legend(title='Product ID')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/product_seasonality_trend.png')
    plt.close()

# In-store vs Online Sales Comparison
def plot_channel_sales_comparison():
    if 'sales_channel' not in sales_data.columns:
        print("Column 'sales_channel' not found in sales_data.")
        return

    channel_sales = sales_data.groupby('sales_channel')['total_sales'].sum().reset_index()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=channel_sales, x='sales_channel', y='total_sales', palette='Set2')
    plt.title('Total Sales by Channel')
    plt.xlabel('Sales Channel')
    plt.ylabel('Total Sales')
    plt.tight_layout()
    plt.savefig('results/channel_sales_comparison.png')
    plt.close()

# Peak hours by channel
def plot_peak_hours_by_channel():
    if 'sales_channel' not in sales_data.columns or 'transaction_date' not in sales_data.columns:
        print("Required columns not found in sales_data.")
        return

    sales_data['hour'] = sales_data['transaction_date'].dt.hour
    hourly_sales = sales_data.groupby(['sales_channel', 'hour'])['total_sales'].sum().reset_index()

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=hourly_sales, x='hour', y='total_sales', hue='sales_channel', marker='o')
    plt.title('Sales by Hour and Channel')
    plt.xlabel('Hour of Day')
    plt.ylabel('Total Sales')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/hourly_channel_sales.png')
    plt.close()

# Top products by channel
def plot_top_products_by_channel(top_n=5):
    if 'sales_channel' not in sales_data.columns or 'product_id' not in sales_data.columns:
        print("Required columns not found in sales_data.")
        return

    top_products_df = sales_data.groupby(['sales_channel', 'product_id'])['total_sales'].sum().reset_index()
    top_products_df = top_products_df.sort_values(['sales_channel', 'total_sales'], ascending=[True, False])

    for channel in top_products_df['sales_channel'].unique():
        top_channel_df = top_products_df[top_products_df['sales_channel'] == channel].head(top_n)

        plt.figure(figsize=(8, 6))
        sns.barplot(data=top_channel_df, x='product_id', y='total_sales', palette='pastel')
        plt.title(f'Top {top_n} Products - {channel}')
        plt.xlabel('Product ID')
        plt.ylabel('Total Sales')
        plt.tight_layout()
        plt.savefig(f'results/top_products_{channel.lower()}.png')
        plt.close()

# Run all visualizations
if __name__ == "__main__":
    plot_sales_trend()
    plot_spending_vs_transactions()
    plot_churn_rate_by_age(customer_df)

    # Dummy example – Replace with real values
    y_true = [0, 1, 0, 1, 0, 1]
    y_pred = [0, 1, 1, 1, 0, 0]
    labels = [0, 1]
    plot_confusion_matrix_custom(y_true, y_pred, labels)

    plot_correlation_heatmap(sales_data)
    plot_sales_histogram()
    plot_sales_distribution(sales_data)
    plot_sales_by_category(sales_data, category_col='store_state', sales_col='total_sales')
    plot_radar_chart()
    plot_product_demand_analysis()
    plot_seasonal_sales_trends()
    plot_product_seasonality()
    plot_channel_sales_comparison()
    plot_peak_hours_by_channel()
    plot_top_products_by_channel()
