import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import os

# Ensure the results directory exists
if not os.path.exists("results"):
    os.makedirs("results")

# Price Sensitivity & Discount Effectiveness Analysis
def analyze_price_sensitivity_and_discount(cleaned_transaction_df):
    print("Running Price Sensitivity & Discount Effectiveness Analysis...")

    # Select relevant columns (price, quantity, and discount)
    transaction_df = cleaned_transaction_df.toPandas()

    # Price Sensitivity: Correlation between price and sales
    transaction_df['total_sales'] = transaction_df['quantity'] * transaction_df['unit_price']
    price_sales_corr = transaction_df[['unit_price', 'total_sales']].corr().iloc[0, 1]

    # Discount Effectiveness: Correlation between discount and sales
    transaction_df['discount_percentage'] = transaction_df['discount_applied'] / transaction_df['unit_price'] * 100
    discount_sales_corr = transaction_df[['discount_percentage', 'total_sales']].corr().iloc[0, 1]

    # Model Price Sensitivity (Price vs Sales)
    X_price = transaction_df[['unit_price']]
    y_sales = transaction_df['total_sales']
    price_model = LinearRegression()
    price_model.fit(X_price, y_sales)
    transaction_df['predicted_sales_price'] = price_model.predict(X_price)

    # Model Discount Effectiveness (Discount % vs Sales)
    X_discount = transaction_df[['discount_percentage']]
    discount_model = LinearRegression()
    discount_model.fit(X_discount, y_sales)
    transaction_df['predicted_sales_discount'] = discount_model.predict(X_discount)

    # Visualize Price Sensitivity
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='unit_price', y='total_sales', data=transaction_df, color='blue')
    plt.plot(transaction_df['unit_price'], transaction_df['predicted_sales_price'], color='red', label='Price Sensitivity Line')
    plt.title('Price Sensitivity: Sales vs Price')
    plt.xlabel('Unit Price')
    plt.ylabel('Total Sales')
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/price_sensitivity.png")
    plt.close()

    # Visualize Discount Effectiveness
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='discount_percentage', y='total_sales', data=transaction_df, color='green')
    plt.plot(transaction_df['discount_percentage'], transaction_df['predicted_sales_discount'], color='red', label='Discount Effectiveness Line')
    plt.title('Discount Effectiveness: Sales vs Discount Percentage')
    plt.xlabel('Discount Percentage')
    plt.ylabel('Total Sales')
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/discount_effectiveness.png")
    plt.close()

    # Save results (correlations and models)
    results = {
        "price_sales_correlation": price_sales_corr,
        "discount_sales_correlation": discount_sales_corr
    }

    # Save the results in a CSV
    pd.DataFrame([results]).to_csv("results/price_discount_analysis.csv", index=False)

    print("Price Sensitivity & Discount Effectiveness Analysis completed.\nResults saved.")
