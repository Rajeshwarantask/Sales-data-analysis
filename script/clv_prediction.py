import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure the results directory exists
if not os.path.exists("results"):
    os.makedirs("results")

# CLV Prediction Function
def predict_clv(enriched_customer_df):
    print("Running Customer Lifetime Value (CLV) Prediction...")

    # Select relevant columns for CLV prediction (RFM features)
    rfm_df = enriched_customer_df.select(
        'customer_id', 'customer_transactions_count', 'customer_average_spent', 'customer_days_since_last_purchase'
    ).toPandas()
    # Print the size of the DataFrame
    print("Number of records in the dataset:", rfm_df.shape[0])  # Rows
    print("Number of features in the dataset:", rfm_df.shape[1])  # Columns

    # Model to predict CLV (using frequency and monetary value as features)
    X = rfm_df[['customer_transactions_count', 'customer_average_spent', 'customer_days_since_last_purchase']]
    y = rfm_df['customer_average_spent']  # Assuming 'customer_average_spent' as target for simplicity

    # Train a regression model (Linear Regression for simplicity)
    model = LinearRegression()
    model.fit(X, y)

    # Predict CLV
    rfm_df['predicted_clv'] = model.predict(X)

    # Visualizing the CLV distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(rfm_df['predicted_clv'], kde=True, color='blue')
    plt.title('Customer Lifetime Value Distribution')
    plt.xlabel('Predicted CLV')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig("results/clv_distribution.png")
    plt.close()

    # Visualizing the top N high-value customers (top 10 for example)
    top_n_customers = rfm_df.nlargest(10, 'predicted_clv')
    plt.figure(figsize=(10, 6))
    sns.barplot(x='customer_id', y='predicted_clv', data=top_n_customers)
    plt.title('Top 10 High-Value Customers')
    plt.xlabel('Customer ID')
    plt.ylabel('Predicted CLV')
    plt.tight_layout()
    plt.savefig("results/top_10_customers_clv.png")
    plt.close()

    # Save the CLV prediction results
    rfm_df.to_csv("results/clv_predictions.csv", index=False)

    print("Customer Lifetime Value (CLV) prediction completed.\nCLV distribution and top customers saved.")
