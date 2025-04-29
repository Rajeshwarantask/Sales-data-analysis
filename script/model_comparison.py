import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.dummy import DummyClassifier, DummyRegressor
from xgboost import XGBClassifier, XGBRegressor
from prophet import Prophet
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# -------------------- Data Load --------------------
df = pd.read_csv('C:\Users\Lenovo\OneDrive\Desktop\sales big data\data\retail_data.csv')
os.makedirs("results", exist_ok=True)

# -------------------- Preprocessing --------------------
imputer = KNNImputer(n_neighbors=5)
df[df.columns] = imputer.fit_transform(df)

def extract_date_features(df, date_col='transaction_date'):
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df[f'{date_col}_year'] = df[date_col].dt.year
    df[f'{date_col}_month'] = df[date_col].dt.month
    df[f'{date_col}_dayofweek'] = df[date_col].dt.dayofweek
    return df

df = extract_date_features(df)

X = df.drop(['churn', 'sales_amount', 'transaction_date'], axis=1)
y_class = df['churn']
y_reg = df['sales_amount']

# -------------------- Train/Test Split --------------------
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42, stratify=y_class)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)

# -------------------- Scaling --------------------
scaler = StandardScaler()
X_train_c = scaler.fit_transform(X_train_c)
X_test_c = scaler.transform(X_test_c)
X_train_r = scaler.fit_transform(X_train_r)
X_test_r = scaler.transform(X_test_r)

# -------------------- Classification Models --------------------
def evaluate_classification_models(X_train, X_test, y_train, y_test):
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    models = {
        'Dummy Classifier': DummyClassifier(strategy='most_frequent'),
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=200, class_weight='balanced'),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        'Gradient Boosting': GradientBoostingClassifier(),
        'SVM': SVC(probability=True, class_weight='balanced')
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred

        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision,
            'Recall': recall,
            'F1-score': f1,
            'ROC AUC': roc_auc_score(y_test, y_proba),
            'Confusion Matrix': confusion_matrix(y_test, y_pred)
        }
    return results

# -------------------- Regression Models --------------------
def prepare_prophet_data(df_subset):
    return pd.DataFrame({'ds': pd.to_datetime(df_subset['transaction_date']), 'y': df_subset['sales_amount']})

def evaluate_regression_models(X_train, X_test, y_train, y_test, df_train, df_test):
    models = {
        'Dummy Regressor': DummyRegressor(strategy='mean'),
        'Linear Regression': LinearRegression(),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=200),
        'XGBoost Regressor': XGBRegressor()
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R²': r2_score(y_test, y_pred)
        }

    try:
        prophet_train = prepare_prophet_data(df_train)
        prophet_test = prepare_prophet_data(df_test)
        prophet_model = Prophet()
        prophet_model.fit(prophet_train)
        forecast = prophet_model.predict(prophet_test[['ds']])
        y_pred = forecast['yhat'].values
        results['Prophet'] = {
            'MAE': mean_absolute_error(prophet_test['y'], y_pred),
            'RMSE': np.sqrt(mean_squared_error(prophet_test['y'], y_pred)),
            'R²': r2_score(prophet_test['y'], y_pred)
        }
    except Exception as e:
        print(f"[Warning] Prophet failed: {e}")

    return results

# -------------------- Plots & Print --------------------
def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"results/{model_name}_conf_matrix.png")
    plt.close()

def print_results(results, title):
    print(f"\n{title}:\n" + "="*len(title))
    for model, metrics in results.items():
        print(f"\n{model}:")
        for key, val in metrics.items():
            print(f"  {key}: {val if not isinstance(val, float) else round(val, 4)}")

# -------------------- Run All --------------------
classification_results = evaluate_classification_models(X_train_c, X_test_c, y_train_c, y_test_c)
regression_results = evaluate_regression_models(
    X_train_r, X_test_r, y_train_r, y_test_r,
    df.loc[X_train_r.shape[0]:], df.loc[X_test_r.shape[0]:]
)

print_results(classification_results, "Churn Prediction Results")
for model, metrics in classification_results.items():
    plot_confusion_matrix(metrics['Confusion Matrix'], model)

print_results(regression_results, "Sales Forecasting Results")

pd.DataFrame(classification_results).T.to_csv("results/churn_model_comparison.csv")
pd.DataFrame(regression_results).T.to_csv("results/sales_model_comparison.csv")

# -------------------- Grid Search Optimization --------------------
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid_search = GridSearchCV(
    RandomForestClassifier(class_weight='balanced'),
    param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='f1', n_jobs=-1, verbose=1
)
grid_search.fit(X_train_c, y_train_c)
print(f"\nBest Random Forest F1 Score: {grid_search.best_score_:.4f}")
print(f"Best Parameters: {grid_search.best_params_}")
