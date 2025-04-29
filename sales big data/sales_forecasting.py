from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, year, month, to_date, sum as _sum

def run_forecasting(transaction_df: DataFrame):
    # Convert transaction_date to DateType
    transaction_df = transaction_df.withColumn("transaction_date", to_date(col("transaction_date"), "MM-dd-yyyy"))

    # Add year and month columns
    transaction_df = transaction_df.withColumn("year", year(col("transaction_date"))) \
                                     .withColumn("month", month(col("transaction_date")))

    # Ensure quantity and unit_price are numeric
    transaction_df = transaction_df.withColumn("quantity", col("quantity").cast("float")) \
                                     .withColumn("unit_price", col("unit_price").cast("float"))

    # Calculate sales_amount
    transaction_df = transaction_df.withColumn("sales_amount", col("quantity") * col("unit_price"))

    # Check initial data
    print("Initial data count:", transaction_df.count())
    transaction_df.show(5)

    # Check for nulls in relevant columns
    null_counts = transaction_df.select(
        (col("year").isNull().cast("int").alias("year_nulls")),
        (col("month").isNull().cast("int").alias("month_nulls")),
        (col("sales_amount").isNull().cast("int").alias("sales_amount_nulls"))
    )

    # Aggregate the null counts
    null_counts = null_counts.agg(
        _sum("year_nulls").alias("total_year_nulls"),
        _sum("month_nulls").alias("total_month_nulls"),
        _sum("sales_amount_nulls").alias("total_sales_amount_nulls")
    )

    null_counts.show()

    # Remove rows with null values
    transaction_df = transaction_df.na.drop(subset=["year", "month", "sales_amount"])
    print("Count after dropping nulls:", transaction_df.count())

    # Assemble features for linear regression
    assembler = VectorAssembler(inputCols=["year", "month"], outputCol="features")
    transaction_df = assembler.transform(transaction_df)

    # Train-test split
    train_df, test_df = transaction_df.randomSplit([0.8, 0.2], seed=42)

    # Check if train_df is empty
    print("Training data count:", train_df.count())
    if train_df.count() == 0:
        print("Training dataset is empty. Check your data processing steps.")
        return

    # Linear Regression Model
    lr = LinearRegression(featuresCol="features", labelCol="sales_amount", regParam=0.1)
    lr_model = lr.fit(train_df)

    # Predict and evaluate
    predictions = lr_model.transform(test_df)
    predictions.select("prediction", "sales_amount").show(5)

    # Print model summary for insights
    print("Coefficients:", lr_model.coefficients)
    print("Intercept:", lr_model.intercept)

    return predictions
