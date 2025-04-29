from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import silhouette_score
import pandas as pd

# Regression Metrics
def evaluate_regression_model(y_true, y_pred):
    """
    Evaluate regression model performance using common metrics.
    Args:
    - y_true: Actual values
    - y_pred: Predicted values
    
    Returns:
    - Dictionary with MAE, MSE, RMSE, and R2 Score.
    """
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred, squared=False),
        "R2 Score": r2_score(y_true, y_pred)
    }

# Classification Metrics
def evaluate_classification_model(y_true, y_pred):
    """
    Evaluate classification model performance using common metrics.
    Args:
    - y_true: Actual labels
    - y_pred: Predicted labels
    
    Returns:
    - Dictionary with Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
    """
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average='binary', zero_division=1),
        "Recall": recall_score(y_true, y_pred, average='binary', zero_division=1),
        "F1-Score": f1_score(y_true, y_pred, average='binary', zero_division=1),
        "Confusion Matrix": confusion_matrix(y_true, y_pred).tolist()
    }

# Clustering Metrics
def evaluate_clustering_model(features, labels):
    """
    Evaluate clustering model performance using Silhouette Score.
    Args:
    - features: The data points used for clustering
    - labels: The predicted labels (clusters)
    
    Returns:
    - Silhouette Score
    """
    return silhouette_score(features, labels)

# Function to save metrics to CSV
def save_metrics_to_csv(metrics_dict, file_path):
    """
    Save model evaluation metrics to a CSV file.
    Args:
    - metrics_dict: Dictionary containing the model name and corresponding metrics
    - file_path: Path to save the CSV file
    
    Returns:
    - None
    """
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df.to_csv(file_path, index=False)
    print(f"Metrics saved to {file_path}")
