#C:\Users\Lenovo\OneDrive\Desktop\sales big data\script\product_demand_analysis.py
from pyspark.sql import SparkSession
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import VectorAssembler

spark = SparkSession.builder.appName("ProductDemandAnalysis").getOrCreate()

# Load data
transaction_df = spark.read.parquet('data/cleaned_transaction_data.parquet')

# Assemble features
assembler = VectorAssembler(inputCols=["price", "discount", "customer_age"], outputCol="features")
data = assembler.transform(transaction_df)

# Train decision tree model to predict product demand
dt = DecisionTreeClassifier(featuresCol="features", labelCol="high_demand_product")
train_data, test_data = data.randomSplit([0.8, 0.2])

model = dt.fit(train_data)
predictions = model.transform(test_data)

predictions.select("features", "high_demand_product", "prediction").show(5)
