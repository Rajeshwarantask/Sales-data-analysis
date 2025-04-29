import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score

os.makedirs("results", exist_ok=True)

# Load dataset
df = pd.read_csv('C:/Users/Lenovo/OneDrive/Desktop/sales big data/data/retail_data.csv')
df.drop(columns=['transaction_date'], inplace=True, errors='ignore')

# Define targets
target_churn = 'churn'
target_sales = 'sales_amount'

# Define features
X = df.drop(columns=[target_churn, target_sales])
y_churn = df[target_churn]
y_sales = df[target_sales]

# Split features
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Preprocessing
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numerical_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

# PCA pipeline (optional)
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('pca', PCA(n_components=0.95))
])

X_preprocessed = full_pipeline.fit_transform(X)

# Split the dataset
X_train, X_test, y_train_churn, y_test_churn, y_train_sales, y_test_sales = train_test_split(
    X_preprocessed, y_churn, y_sales, test_size=0.2, random_state=42
)

# Model tuning
def tune_model(model, param_grid, X_train, y_train, search_type='grid'):
    search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=1) if search_type == 'grid' else \
             RandomizedSearchCV(model, param_grid, n_iter=50, cv=5, n_jobs=-1, verbose=1, random_state=42)
    search.fit(X_train, y_train)
    print(f"Best parameters for {model.__class__.__name__}: {search.best_params_}")
    return search.best_estimator_

# Hyperparameter grids
rf_clf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

xgb_clf_params = {
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200],
    'max_depth': [3, 6],
    'subsample': [0.8, 1.0]
}

svm_params = {
    'C': [1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale']
}

# Classification models
rf_clf = tune_model(RandomForestClassifier(), rf_clf_params, X_train, y_train_churn, 'grid')
xgb_clf = tune_model(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgb_clf_params, X_train, y_train_churn, 'random')
svm_clf = tune_model(SVC(probability=True), svm_params, X_train, y_train_churn, 'grid')

# Regression models
rf_reg = tune_model(RandomForestRegressor(), rf_clf_params, X_train, y_train_sales, 'grid')
xgb_reg = tune_model(XGBRegressor(), xgb_clf_params, X_train, y_train_sales, 'random')
gbr_reg = tune_model(GradientBoostingRegressor(), {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5]
}, X_train, y_train_sales, 'grid')

# Evaluation function
def evaluate_model(model, X_test, y_test, task='classification'):
    y_pred = model.predict(X_test)
    if task == 'classification':
        return {'Accuracy': accuracy_score(y_test, y_pred)}
    else:
        return {
            'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R²': r2_score(y_test, y_pred)
        }

# Churn results
results_churn = pd.DataFrame([
    evaluate_model(rf_clf, X_test, y_test_churn, 'classification'),
    evaluate_model(xgb_clf, X_test, y_test_churn, 'classification'),
    evaluate_model(svm_clf, X_test, y_test_churn, 'classification')
], index=['Random Forest', 'XGBoost', 'SVM'])

# Sales prediction results
results_sales = pd.DataFrame([
    evaluate_model(rf_reg, X_test, y_test_sales, 'regression'),
    evaluate_model(xgb_reg, X_test, y_test_sales, 'regression'),
    evaluate_model(gbr_reg, X_test, y_test_sales, 'regression')
], index=['Random Forest', 'XGBoost', 'GradientBoosting'])

# Save results
results_churn.to_csv('results/tuned_churn_model_results.csv')
results_sales.to_csv('results/tuned_sales_model_results.csv')

print("All models tuned and evaluated successfully!")
