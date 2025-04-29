import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, unix_timestamp, to_date, year, month, dayofweek
from pyspark.ml.feature import VectorAssembler, StringIndexer, Imputer  # Corrected import for Imputer
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator

# Initialize Spark session
spark = SparkSession.builder.appName("Sales_EDA_Optimized").getOrCreate()

# Load Cleaned Data
transaction_df = spark.read.parquet(r'C:\Users\Lenovo\OneDrive\Desktop\sales big data\data\cleaned_transaction_data.parquet')
customer_df = spark.read.parquet(r'C:\Users\Lenovo\OneDrive\Desktop\sales big data\data\cleaned_customer_data.parquet')

# Data Preprocessing: Handling Missing Values (using Imputer)
# Impute missing values for numeric columns
imputer = Imputer(inputCols=['quantity', 'unit_price', 'product_variety'], outputCols=['quantity_imputed', 'unit_price_imputed', 'product_variety_imputed'])
transaction_df = imputer.fit(transaction_df).transform(transaction_df)

# Handle Categorical Features: Use StringIndexer for categorical columns
indexer = StringIndexer(inputCol="product_category", outputCol="product_category_index")
transaction_df = indexer.fit(transaction_df).transform(transaction_df)

# Exploratory Data Analysis (Sales Trends)
# Sales by Product Category
sales_by_category = transaction_df.groupBy("product_category").sum("sales_amount").orderBy("sum(sales_amount)", ascending=False)
sales_by_category_df = sales_by_category.toPandas()

# Plotting Sales by Product Category
plt.figure(figsize=(10, 6))
sns.barplot(x='product_category', y='sum(sales_amount)', data=sales_by_category_df)
plt.title("Sales by Product Category")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualization/sales_by_category.png')

# Churn Rate Distribution
churn_rate = customer_df.groupBy("churn").count().toPandas()
churn_rate['churn_rate'] = churn_rate['count'] / churn_rate['count'].sum()

# Plotting Churn Rate
plt.figure(figsize=(7, 5))
sns.barplot(x='churn', y='churn_rate', data=churn_rate)
plt.title("Churn Rate Distribution")
plt.tight_layout()
plt.savefig('visualization/churn_rate.png')

# Average Sales by Customer Lifetime Value (CLV)
avg_sales_by_clv = customer_df.join(transaction_df, "customer_id").groupBy("customer_lifetime_value").agg(
    {"sales_amount": "avg"}).withColumnRenamed("avg(sales_amount)", "average_sales").orderBy("customer_lifetime_value")

avg_sales_by_clv_df = avg_sales_by_clv.toPandas()

# Plotting Average Sales by CLV
plt.figure(figsize=(10, 6))
sns.lineplot(x='customer_lifetime_value', y='average_sales', data=avg_sales_by_clv_df)
plt.title("Average Sales by Customer Lifetime Value (CLV)")
plt.tight_layout()
plt.savefig('visualization/average_sales_by_clv.png')

# Feature Importance (Sales Forecasting)
# Assemble features for sales prediction model
assembler = VectorAssembler(inputCols=['quantity_imputed', 'unit_price_imputed', 'product_variety', 'product_category_index'], outputCol="features")
transaction_df = assembler.transform(transaction_df)

# Train a Random Forest model
rf = RandomForestRegressor(featuresCol="features", labelCol="sales_amount")
model = rf.fit(transaction_df)

# Plotting Feature Importance
feature_importance = model.featureImportances
features = ['quantity', 'unit_price', 'product_variety', 'product_category']

# Plotting feature importance
plt.figure(figsize=(8, 5))
sns.barplot(x=features, y=feature_importance)
plt.title("Feature Importance for Sales Prediction")
plt.tight_layout()
plt.savefig('visualization/feature_importance.png')

# Hyperparameter Tuning & Cross-validation
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 50, 100]) \
    .addGrid(rf.maxDepth, [5, 10, 20]) \
    .build()

evaluator = RegressionEvaluator(labelCol="sales_amount", metricName="rmse")
cv = CrossValidator(estimator=rf, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)

# Apply Cross-validation
cv_model = cv.fit(transaction_df)

# Show all plots
plt.show()
