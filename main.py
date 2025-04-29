from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col
import matplotlib.pyplot as plt
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'  # Set to the number of cores you want to use.

import pandas as pd
print("Current working directory:", os.getcwd())

# Import project modules
import script.data_preprocessing as data_preprocessing
import script.sales_forecasting as sales_forecasting
import script.sales_trends as sales_trends
import script.churn_prediction as churn_prediction
import script.promotion_optimization as promotion_optimization
from script import price_sensitivity_discount_effectiveness as price_sensitivity
import script.seasonality_analysis as seasonality
import script.clv_prediction as clv
from script import customer_segmentation as segmentation
from script.customer_segmentation import run_customer_segmentation
from script import visualizations

# Ensure directories exist
os.makedirs('data', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Initialize Spark session
spark = SparkSession.builder.appName("Sales Data Analysis").getOrCreate()
from pyspark.sql import SparkSession

# Initialize the Spark session
spark = SparkSession.builder.appName("SalesData").getOrCreate()

# Read the Excel file using pandas
pandas_df  = pd.read_csv('C:/Users/Lenovo/OneDrive/Desktop/sales big data/data/retail_data.csv')

# Convert pandas DataFrame to Spark DataFrame
data_df = spark.createDataFrame(pandas_df)

# Show the schema to verify the DataFrame structure
data_df.printSchema()

# Load data using Spark
data_path = r'C:\Users\Lenovo\OneDrive\Desktop\sales big data\data\retail_data.csv'

data_df = spark.read.option("header", "true").csv(r"C:\Users\Lenovo\OneDrive\Desktop\sales big data\data\retail_data.csv")
# Print the schema to check the column names and data types
data_df.printSchema()

# Ensure the data file exists
if not os.path.exists(data_path):
    print(f"Error: Data file '{data_path}' not found.")
    spark.stop()
    exit()

data_df = spark.read.csv(data_path, header=True, inferSchema=True)

# Extract necessary tables
# Extract necessary tables with clearly defined column selections

# Customer information DataFrame
customer_columns = [
    "customer_id", "age", "gender", "income_bracket", "loyalty_program", 
    "membership_years", "churn", "marital_status", "number_of_children", 
    "education_level", "occupation", "transactions_count", "average_spent", 
    "days_since_last_purchase", "email_subscriptions", "app_usage", 
    "website_visits", "social_media_engagement"
]
customer_df = data_df.select(*customer_columns)

# Product information DataFrame
product_columns = [
    "product_id", "product_category", "unit_price", "product_name", 
    "product_brand", "product_rating", "product_review_count", "product_stock", 
    "product_return_rate", "product_size", "product_weight", "product_color", 
    "product_material", "product_manufacture_date", "product_expiry_date", 
    "product_shelf_life"
]
product_df = data_df.select(*product_columns)

# Transaction information DataFrame
transaction_columns = [
    "transaction_id", "transaction_date", "customer_id", "product_id", 
    "quantity", "unit_price", "discount_applied", "payment_method", 
    "store_location", "transaction_hour", "day_of_week", "week_of_year", 
    "month_of_year", "avg_purchase_value", "purchase_frequency", 
    "last_purchase_date", "avg_discount_used", "preferred_store", 
    "online_purchases", "in_store_purchases", "avg_items_per_transaction", 
    "total_returned_items", "total_returned_value", "total_sales", 
    "total_discounts_received", "total_items_purchased", 
    "avg_spent_per_category", "max_single_purchase_value", 
    "min_single_purchase_value"
]
transaction_df = data_df.select(*transaction_columns)

# Store information DataFrame
store_columns = [
    "store_zip_code", "store_city", "store_state", "distance_to_store", 
    "holiday_season", "season", "weekend"
]
store_df = data_df.select(*store_columns)



# Convert categorical to numerical
customer_df = customer_df.withColumn(
    "loyalty_program",
    when(col("loyalty_program") == "Yes", 1).otherwise(0)
)

# Show data samples
print("Sample transaction data:")
transaction_df.show(5)

print("Sample product data:")
product_df.show(5)

# Preprocess data
print("Running preprocessing...")
cleaned_customer_df, cleaned_transaction_df, cleaned_product_df, monthly_sales_trends = data_preprocessing.clean_data(customer_df, transaction_df, product_df)

# Show cleaned samples
print("Cleaned Customer DataFrame:")
cleaned_customer_df.show()

print("Cleaned Transaction DataFrame:")
cleaned_transaction_df.show()

# Run predictive models
print("Running Sales Forecasting...")
sales_forecasting.run_forecasting(cleaned_transaction_df)

print("Running Sales sales trend...")
sales_trends.run_sales_trends_analysis(cleaned_transaction_df)

print("Running Churn Prediction...")
churn_prediction.run_churn_prediction(cleaned_customer_df)

print("Running Promotion Optimization...")
promotion_optimization.run_promotion_optimization(cleaned_customer_df, cleaned_transaction_df)

print("Running Customer Segmentation...")
segmentation.run_customer_segmentation(cleaned_customer_df)

print("Running Seasonality Analysis...")
seasonality.run_seasonality_analysis(cleaned_transaction_df)

print("Running Customer Lifetime Value Prediction...")
clv.predict_clv(cleaned_customer_df)

print("Running Price Sensitivity and Discount Effectiveness Analysis...")
price_sensitivity.analyze_price_sensitivity_and_discount(cleaned_transaction_df)


# Run visualizations (uses pandas, so re-load as needed if required)
print("Generating Visualizations...")
visualizations.plot_sales_trend()
visualizations.plot_spending_vs_transactions()
visualizations.plot_churn_rate_by_age(cleaned_customer_df.toPandas())
visualizations.plot_radar_chart()
visualizations.plot_product_demand_analysis()


# Assuming `sales_df` or `data_df` is your dataset
visualizations.plot_correlation_heatmap(data_df.toPandas())
visualizations.plot_sales_distribution(data_df)
visualizations.plot_sales_by_category(data_df, category_col='region')

# Show any pending matplotlib visuals
plt.show()

# Stop Spark session
spark.stop()

# Log current directory
print("Current working directory:", os.getcwd())
