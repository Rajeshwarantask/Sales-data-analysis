from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import Imputer
from pyspark.sql.types import DoubleType, IntegerType, StringType

# Initialize Spark session with legacy time parser policy
spark = SparkSession.builder \
    .appName("Sales Forecasting") \
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
    .getOrCreate()

# Function to convert date formats
def convert_date_format(df, date_columns):
    for date_col in date_columns:
        if date_col in df.columns:
            df = df.withColumn(
                date_col, 
                F.date_format(F.to_date(date_col, "M-d-yyyy"), "yyyy-MM-dd")
            )
    return df

# Function to impute missing values
def impute_missing_values(df):
    # Convert date columns
    date_columns = ["transaction_date", "promotion_start_date", "promotion_end_date", 
                    "product_manufacture_date", "product_expiry_date"]
    df = convert_date_format(df, date_columns)

    # Handle missing values for numeric columns
    numeric_columns = [c for c in df.columns if df.schema[c].dataType in [DoubleType(), IntegerType()]]
    if numeric_columns:
        imputer = Imputer(inputCols=numeric_columns, outputCols=[f"{c}_imputed" for c in numeric_columns])
        model = imputer.fit(df)
        df = model.transform(df)

    # Fill missing values for categorical columns with "Unknown"
    categorical_columns = [c for c in df.columns if df.schema[c].dataType == StringType()]
    for cat_col in categorical_columns:
        df = df.na.fill("Unknown", subset=[cat_col])

    return df

# Function to clean data
def clean_data(customer_df, transaction_df, product_df):
    # Impute missing values in each DataFrame
    cleaned_customer_df = impute_missing_values(customer_df)
    cleaned_transaction_df = impute_missing_values(transaction_df)
    cleaned_product_df = impute_missing_values(product_df)

    return cleaned_customer_df, cleaned_transaction_df, cleaned_product_df

# Function to compute customer metrics
def compute_customer_metrics(cleaned_transaction_df, cleaned_customer_df):
    # Compute metrics based on transactions
    customer_metrics = cleaned_transaction_df.groupBy("customer_id").agg(
        F.count("transaction_id").alias("transactions_count"),
        F.avg("quantity * unit_price").alias("average_spent"),  # Adjust calculation as necessary
        F.max(F.datediff(F.current_date(), "transaction_date")).alias("days_since_last_purchase")
    )

    # Join these metrics with customer_df
    cleaned_customer_df = cleaned_customer_df.join(customer_metrics, on="customer_id", how="left")

    # Define churn as integer (0 or 1)
    cleaned_customer_df = cleaned_customer_df.withColumn(
        "churn",
        F.when(F.col("days_since_last_purchase") > 30, 1).otherwise(0).cast(IntegerType())  # Cast to Integer
    )

    return cleaned_customer_df

