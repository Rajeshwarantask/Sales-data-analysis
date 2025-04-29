import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load your sales data
sales_data = pd.read_csv('data/retail_data.csv')  # Adjust the path if needed
customer_df = pd.read_csv('data/retail_data.csv')  # Load your customer data if necessary

# Ensure date column is in datetime format
sales_data['transaction_date'] = pd.to_datetime(sales_data['transaction_date'])

# 1. Sales Trend Visualization
def plot_sales_trend():
    sales_trend = sales_data.resample('ME', on='transaction_date').sum()  # Monthly sales
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

# 2. Spending vs Transactions
def plot_spending_vs_transactions():
    plt.figure(figsize=(50, 16))

    # Filter data to focus on a specific range (e.g., transactions_count < 100 and average_spent < 500)
    filtered_data = customer_df[
        (customer_df['transactions_count'] <= 100) & 
        (customer_df['average_spent'] <= 500)
    ]

    # Scatter plot with transparency for clarity
    plt.scatter(
        filtered_data['transactions_count'], 
        filtered_data['average_spent'], 
        alpha=0.5, 
        color='blue'
    )

    # Set meaningful axis limits to avoid overcrowding
    plt.xlim(20, 100)  # Adjust x-axis range as needed
    plt.ylim(100, 600)  # Adjust y-axis range as needed

    # Add labels and title
    plt.title('Customer Spending vs. Transactions Count')
    plt.xlabel('Transactions Count')
    plt.ylabel('Average Spending')

    # Save and close the plot
    plt.savefig('results/spending_vs_transactions.png')
    plt.close()

# 3. Churn Rate by Age
def plot_churn_rate_by_age():
    # Optional: Sample data to avoid overcrowding if dataset is large
    sample_size = 100000  # Adjust the sample size as needed
    
    # Ensure customer_df is initialized properly
    global customer_df

    # Check if we need to sample the data based on size
    if len(customer_df) > sample_size:
        customer_df = customer_df.sample(n=sample_size, random_state=42)  # Sampling a subset of data

    plt.figure(figsize=(20, 16))
    
    # Create a countplot for churn rate by age
    sns.countplot(data=customer_df, x='age', hue='churn')
    
    # Set title and labels
    plt.title('Churn Rate by Age')
    plt.xlabel('Age')
    plt.ylabel('Count')

    # Set custom x and y axis ranges
    plt.xticks(np.arange(18, 82, 2))  # Age increments of 2 (18, 20, ..., 80)
    plt.yticks(np.arange(0, 11000, 2000))  # Count increments of 2000 (2000, 4000, ..., 10000)  # Adjust y-axis range (example: count between 0 and 5000)

    # Add legend
    plt.legend(title='Churn', loc='upper right', labels=['Not Churned', 'Churned'])
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Add grid for clarity
    plt.grid()

    # Save the plot
    plt.tight_layout()
    plt.savefig('results/churn_rate_by_age.png')
    plt.close()

# 4. Sales Distribution
def plot_sales_distribution():
    plt.figure(figsize=(26, 16))
    plt.hist(sales_data['total_sales'], bins=30, edgecolor='k', alpha=0.7)
    plt.title('Sales Amount Distribution')
    plt.xlabel('Sales Amount')
    plt.ylabel('Frequency')
    plt.grid()
    plt.savefig('results/sales_distribution.png')
    plt.close()

# 5. Radar Chart for Sales by Product Category
def plot_radar_chart():
    categories = list(customer_df['product_category'].unique())
    values = [customer_df[customer_df['product_category'] == category]['total_sales'].sum() for category in categories]
    
    # Number of variables
    num_vars = len(categories)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # The plot is a circle, so we need to "complete the loop" and append the start value to the end.
    values += values[:1]
    angles += angles[:1]

    # Draw the radar chart
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='red', alpha=0.25)
    ax.plot(angles, values, color='red', linewidth=2)

    # Labels for each axis
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)

    plt.title('Sales by Product Category')
    plt.savefig('results/sales_by_product_category.png')
    plt.close()

# 6. Product Demand Analysis with Predictions
def plot_product_demand_analysis():
    # Add 'month_year' column for aggregation
    sales_data['month_year'] = pd.to_datetime(sales_data['transaction_date']).dt.to_period('M')
    
    # Group by month and aggregate relevant columns
    demand_trend = sales_data.groupby('month_year').agg({
        'total_sales': 'sum',
        'discount_applied': 'mean',  # Adjusted for your dataset
        'unit_price': 'mean',       # Adjusted for your dataset
    }).reset_index()
    
    # Convert 'month_year' back to timestamp for plotting
    demand_trend['month_year'] = demand_trend['month_year'].dt.to_timestamp()
    
    # Add a time index for regression
    demand_trend['time_index'] = np.arange(len(demand_trend))
    
    # Use NumPy for polynomial regression
    X = demand_trend['time_index'].values
    y = demand_trend['total_sales'].values
    coefficients = np.polyfit(X, y, 3)  # Polynomial regression (degree 3)
    regression_line = np.polyval(coefficients, X)  # Compute predictions
    
    # Plot actual and predicted sales trends
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

# Run all visualizations
if __name__ == "__main__":
    plot_sales_trend()
    plot_spending_vs_transactions()
    plot_churn_rate_by_age()
    plot_sales_distribution()
    plot_radar_chart()  # Add this line to run the radar chart
    plot_product_demand_analysis()
