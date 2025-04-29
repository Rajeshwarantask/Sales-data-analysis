from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def run_churn_prediction(customer_df):
    # Before assembling features
    print("Available columns in customer_df:", customer_df.columns)
    customer_df.printSchema()

    # Check for required columns
    required_columns = ["transactions_count", "average_spent", "days_since_last_purchase", "churn"]
    missing_columns = [col for col in required_columns if col not in customer_df.columns]
    
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        return

    # Assemble features
    assembler = VectorAssembler(inputCols=["transactions_count", "average_spent", "days_since_last_purchase"], outputCol="features")
    data = assembler.transform(customer_df)

    # Train logistic regression model for churn prediction
    lr = LogisticRegression(featuresCol="features", labelCol="churn")
    train_data, test_data = data.randomSplit([0.8, 0.2])

    model = lr.fit(train_data)
    
    # Make predictions
    predictions = model.transform(test_data)
    predictions.select("prediction", "churn").show(5)

    # Evaluate the model
    evaluator = MulticlassClassificationEvaluator(labelCol="churn", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print(f"Accuracy of the model: {accuracy:.2f}")

    return predictions
