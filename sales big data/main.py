from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col
import pandas as pd  # Import pandas if you need it later
import data_preprocessing
import sales_forecasting
import churn_prediction
import promotion_optimization

# Initialize Spark session
spark = SparkSession.builder.appName("Sales Data Analysis").getOrCreate()

# Load retail data using Spark
data_df = spark.read.csv('C:/Users/Lenovo/OneDrive/Desktop/sales big data/data/retail_data.csv', header=True, inferSchema=True)

# Create separate DataFrames from Spark DataFrame
customer_df = data_df.select("customer_id", "age", "loyalty_program","days_since_last_purchase","churn","transactions_count","average_spent")  # Adjust as necessary
product_df = data_df.select("product_id", "product_category", "unit_price")  # Adjust as necessary
transaction_df = data_df.select("transaction_id", "transaction_date", "product_id", "quantity", "unit_price")  # Adjust as necessary

# Convert loyalty_program to numeric in Spark DataFrame
customer_df = customer_df.withColumn(
    "loyalty_program",
    when(col("loyalty_program") == "Yes", 1).otherwise(0)
)

# Show transaction and product DataFrames
transaction_df.show(5)
product_df.show(5)

# Run preprocessing script
cleaned_customer_df, cleaned_transaction_df, cleaned_product_df = data_preprocessing.clean_data(customer_df, transaction_df, product_df)

# Check the cleaned DataFrames
print("Cleaned Customer DataFrame:")
cleaned_customer_df.show()

print("Cleaned Transaction DataFrame:")
cleaned_transaction_df.show()

# Call the other scripts
print("Running Sales Forecasting...")
sales_forecasting.run_forecasting(cleaned_transaction_df)

print("Running Churn Prediction...")
churn_prediction.run_churn_prediction(cleaned_customer_df)

print("Running Promotion Optimization...")
promotion_optimization.run_promotion_optimization(cleaned_customer_df, cleaned_transaction_df)

print("All tasks completed!")

# Stop Spark session
spark.stop()

# Optional: Print current working directory
import os
print(os.getcwd())

# If using matplotlib, ensure to import before using plt.show()
import matplotlib.pyplot as plt
plt.show()
