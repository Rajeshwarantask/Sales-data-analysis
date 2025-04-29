from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler

def run_promotion_optimization(customer_df, transaction_df):
    # Feature engineering
    assembler = VectorAssembler(inputCols=["transactions_count", "average_spent"], outputCol="features")
    data = assembler.transform(customer_df)

    # KMeans clustering to segment customers for targeted promotions
    kmeans = KMeans(featuresCol="features", k=5)
    model = kmeans.fit(data)

    # Predict customer segments
    predictions = model.transform(data)
    predictions.select("customer_id", "prediction").show(5)
