from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import Imputer
from pyspark.sql.types import DoubleType, IntegerType, StringType, DateType
from prophet import Prophet
import pandas as pd

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Sales Forecasting Enhanced") \
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


# Compute customer metrics
def compute_customer_metrics(cleaned_transaction_df, cleaned_customer_df):
    cleaned_transaction_df = cleaned_transaction_df.withColumn(
        "total_spent", cleaned_transaction_df["quantity"] * cleaned_transaction_df["unit_price"]
    )

    customer_metrics = cleaned_transaction_df.groupBy("customer_id").agg(
        F.count("transaction_id").alias("customer_transactions_count"),
        F.avg("total_spent").alias("customer_average_spent"),
        F.max(F.datediff(F.current_date(), "transaction_date")).alias("customer_days_since_last_purchase")
    )

    enriched_customer_df = cleaned_customer_df.join(customer_metrics, on="customer_id", how="left")

    enriched_customer_df = enriched_customer_df.withColumn(
        "churn",
        F.when(F.col("customer_days_since_last_purchase") > 30, 1).otherwise(0).cast(IntegerType())
    )

    return enriched_customer_df


# Forecast monthly sales using Prophet
def compute_monthly_sales_trends(cleaned_transaction_df):
    monthly_sales = cleaned_transaction_df.withColumn(
        "month", F.date_format("transaction_date", "yyyy-MM")
    ).groupBy("month").agg(
        F.sum(F.col("quantity") * F.col("unit_price")).alias("total_sales")
    ).orderBy("month")

    # Convert to Pandas for forecasting
    monthly_sales_pd = monthly_sales.toPandas()
    monthly_sales_pd.rename(columns={"month": "ds", "total_sales": "y"}, inplace=True)
    monthly_sales_pd["ds"] = pd.to_datetime(monthly_sales_pd["ds"], format="%Y-%m")

    # Prophet model
    model = Prophet()
    model.fit(monthly_sales_pd)

    # Forecast 6 future months
    future = model.make_future_dataframe(periods=6, freq="M")
    forecast = model.predict(future)

    # Select useful columns
    forecast_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    print("\n--- Forecasted Monthly Sales (Next 6 Months) ---")
    print(forecast_df.tail(6))

    # Return original monthly sales as Spark DataFrame
    return spark.createDataFrame(monthly_sales_pd)


# Average basket size per transaction
def compute_average_basket_size(cleaned_transaction_df):
    avg_basket = cleaned_transaction_df.groupBy("transaction_id").agg(
        (F.sum("quantity") * F.sum("unit_price")).alias("basket_value")
    )
    avg = avg_basket.agg(F.avg("basket_value").alias("average_basket_size")).first()["average_basket_size"]
    print(f"\nAverage Basket Size: {avg:.2f}")
    return cleaned_transaction_df


# Extract seasonal trends
def compute_seasonal_trends(cleaned_transaction_df):
    cleaned_transaction_df = cleaned_transaction_df.withColumn(
        "season", F.month("transaction_date")
    )
    return cleaned_transaction_df


# Optional: Product metrics
def compute_product_metrics(cleaned_transaction_df):
    return cleaned_transaction_df.groupBy("product_id").agg(
        F.sum("quantity").alias("total_quantity_sold"),
        F.sum(F.col("quantity") * F.col("unit_price")).alias("total_revenue"),
        F.avg("unit_price").alias("average_price")
    )


# Clean all datasets
def clean_data(customer_df, transaction_df, product_df):
    cleaned_customer_df = impute_missing_values(customer_df)
    cleaned_transaction_df = impute_missing_values(transaction_df)
    cleaned_product_df = impute_missing_values(product_df)

    enriched_customer_df = compute_customer_metrics(cleaned_transaction_df, cleaned_customer_df)
    cleaned_transaction_df = compute_average_basket_size(cleaned_transaction_df)
    cleaned_transaction_df = compute_seasonal_trends(cleaned_transaction_df)

    monthly_sales_trends = compute_monthly_sales_trends(cleaned_transaction_df)

    return enriched_customer_df, cleaned_transaction_df, cleaned_product_df, monthly_sales_trends


# Example usage:
# customer_df, transaction_df, product_df = load_data()  # Load your DataFrames here
# enriched_customer_df, cleaned_transaction_df, cleaned_product_df, monthly_sales_trends = clean_data(customer_df, transaction_df, product_df)
