import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Sales_EDA").getOrCreate()

# Load Cleaned Data
transaction_df = spark.read.parquet(r'C:\Users\Lenovo\OneDrive\Desktop\sales big data\data\cleaned_transaction_data.parquet')



# Exploratory Data Analysis (Sales Trends)
sales_by_category = transaction_df.groupBy("product_category").sum("sales_amount").orderBy("sum(sales_amount)", ascending=False)
sales_by_category_df = sales_by_category.toPandas()

# Plotting
plt.figure(figsize=(10, 6))
sns.barplot(x='product_category', y='sum(sales_amount)', data=sales_by_category_df)
plt.title("Sales by Product Category")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualization/sales_by_category.png')
