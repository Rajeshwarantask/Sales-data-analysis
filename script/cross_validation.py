import pandas as pd
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from imblearn.over_sampling import SMOTE
from pyspark.ml.evaluation import RegressionEvaluator

# Initialize Spark session
spark = SparkSession.builder.appName("ImprovedCrossValidation").getOrCreate()

# Load Data
transaction_df = spark.read.parquet('data/cleaned_transaction_data.parquet')

# Feature engineering: Assemble features
assembler = VectorAssembler(inputCols=["price", "discount", "customer_age"], outputCol="features")
data = assembler.transform(transaction_df)

# Scale features
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
data = scaler.fit(data).transform(data)

# Handle class imbalance using SMOTE
def smote_oversample(data, label_col, feature_col):
    # Convert Spark DataFrame to Pandas for SMOTE processing
    pandas_df = data.select(label_col, feature_col).toPandas()
    smote = SMOTE(sampling_strategy='auto')
    X_resampled, y_resampled = smote.fit_resample(pandas_df[feature_col], pandas_df[label_col])
    
    # Convert back to Spark DataFrame
    return spark.createDataFrame(pd.DataFrame(X_resampled, columns=feature_col), schema=data.schema)

# Oversample data
data = smote_oversample(data, label_col="high_demand_product", feature_col=["scaled_features"])

# Define walk-forward validation function
def walk_forward_validation(data, model, param_grid, evaluator, num_folds=10):
    n = data.count()
    fold_size = int(n / num_folds)
    
    best_model = None
    best_score = float('inf')  # We aim for minimization (e.g., MAE, RMSE)
    
    # Loop through the walk-forward validation steps
    for fold in range(num_folds):
        train_data = data.limit(fold * fold_size)
        test_data = data.limit((fold + 1) * fold_size).subtract(train_data)
        
        # Build the cross-validator
        cross_val = CrossValidator(estimator=model,
                                   estimatorParamMaps=param_grid,
                                   evaluator=evaluator,
                                   numFolds=3)  # 3-fold within each validation set
        
        # Train the model with cross-validation
        cv_model = cross_val.fit(train_data)
        
        # Evaluate the model on the test set
        predictions = cv_model.bestModel.transform(test_data)
        score = evaluator.evaluate(predictions)
        
        print(f"Fold {fold + 1}, Score (AUC-ROC): {score}")
        
        # Check if this model is better
        if score < best_score:
            best_score = score
            best_model = cv_model.bestModel
    
    return best_model

# Example models: Logistic Regression, Random Forest, Gradient Boosting
lr = LogisticRegression(featuresCol="scaled_features", labelCol="high_demand_product")
rf = RandomForestClassifier(featuresCol="scaled_features", labelCol="high_demand_product")
gbt = GBTClassifier(featuresCol="scaled_features", labelCol="high_demand_product")

# Define param grids for each model with expanded ranges
lr_param_grid = ParamGridBuilder().addGrid(lr.regParam, [0.01, 0.1, 0.5]).addGrid(lr.maxIter, [10, 20, 50]).build()
rf_param_grid = ParamGridBuilder().addGrid(rf.numTrees, [50, 100, 200]).addGrid(rf.maxDepth, [5, 10, 15]).build()
gbt_param_grid = ParamGridBuilder().addGrid(gbt.maxIter, [10, 50, 100]).addGrid(gbt.maxDepth, [5, 10, 15]).build()

# Define evaluators
binary_evaluator = BinaryClassificationEvaluator(labelCol="high_demand_product", rawPredictionCol="prediction", metricName="areaUnderROC")

# Perform walk-forward validation for each model
lr_best_model = walk_forward_validation(data, lr, lr_param_grid, binary_evaluator)
rf_best_model = walk_forward_validation(data, rf, rf_param_grid, binary_evaluator)
gbt_best_model = walk_forward_validation(data, gbt, gbt_param_grid, binary_evaluator)

# Output the best model parameters and performance
print("Best Logistic Regression Model Parameters:")
print(lr_best_model.extractParamMap())
print("Best Random Forest Model Parameters:")
print(rf_best_model.extractParamMap())
print("Best Gradient Boosting Model Parameters:")
print(gbt_best_model.extractParamMap())

# Stop the Spark session
spark.stop()
