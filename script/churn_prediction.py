import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import joblib
import os
from script.visualizations import plot_confusion_matrix_custom
from script.metrics_utils import evaluate_classification_model

def run_churn_prediction(cleaned_customer_df):
    print("Preparing churn prediction...")

    # Ensure the results directory exists
    if not os.path.exists("results"):
        os.makedirs("results")

    # Convert to Pandas DataFrame
    customer_pd = cleaned_customer_df.toPandas()

    # Drop rows with nulls in key columns
    customer_pd = customer_pd.dropna(subset=["churn", "transactions_count", "average_spent"])

    # Feature matrix and label
    X = customer_pd[["age", "loyalty_program", "days_since_last_purchase", "transactions_count", "average_spent"]]
    y = customer_pd["churn"]

    # Check if churn has both classes
    if len(y.unique()) < 2:
        print("❌ Not enough class variation in churn labels. At least two classes are required for classification.")
        print("Churn value counts:\n", y.value_counts())
        return

    # Print class distribution
    print("Churn value distribution (raw):\n", customer_pd["churn"].value_counts())

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "SVM": SVC(probability=True),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    results = []
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Evaluate each model
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            metrics = evaluate_classification_model(model, X_test, y_test)
            f1_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='f1')
            metrics["CV F1"] = f1_scores.mean()
            metrics["Model"] = name
            results.append(metrics)
        except Exception as e:
            print(f"Error fitting {name}: {e}")
            continue

    # Hyperparameter tuning (Random Forest)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20]
    }
    rf_grid = GridSearchCV(RandomForestClassifier(), param_grid, scoring='f1', cv=3, n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    rf_metrics = evaluate_classification_model(best_rf, X_test, y_test)
    rf_metrics["CV F1"] = cross_val_score(best_rf, X_scaled, y, cv=cv, scoring='f1').mean()
    rf_metrics["Model"] = "Tuned Random Forest"
    results.append(rf_metrics)

    # Voting Ensemble
    ensemble_model = VotingClassifier(estimators=[ 
        ('logreg', LogisticRegression()),
        ('rf', RandomForestClassifier()),
        ('gb', GradientBoostingClassifier()),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    ], voting='soft')
    ensemble_model.fit(X_train, y_train)
    ensemble_metrics = evaluate_classification_model(ensemble_model, X_test, y_test)
    ensemble_metrics["CV F1"] = cross_val_score(ensemble_model, X_scaled, y, cv=cv, scoring='f1').mean()
    ensemble_metrics["Model"] = "Voting Ensemble"
    results.append(ensemble_metrics)

    # Results summary
    results_df = pd.DataFrame(results).set_index("Model")
    print("\nModel Comparison (Churn Prediction):\n", results_df)
    results_df.to_csv("results/churn_model_comparison.csv")

    # Save best model
    best_model_name = results_df["F1 Score"].idxmax()
    best_model = (
        models.get(best_model_name)
        if best_model_name in models
        else best_rf if best_model_name == "Tuned Random Forest"
        else ensemble_model if best_model_name == "Voting Ensemble"
        else None
    )
    if best_model:
        joblib.dump(best_model, "results/best_churn_model.pkl")

        # Confusion Matrix
        y_pred = best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix_custom(cm, best_model_name, "results/confusion_matrix_churn.png")

        # ROC Curve
        if hasattr(best_model, "predict_proba"):
            probs = best_model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, probs)
            roc_auc = auc(fpr, tpr)

            plt.figure()
            plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {best_model_name}')
            plt.legend(loc='lower right')
            plt.savefig(f"results/roc_curve_{best_model_name}.png")
            plt.show()

    print("✅ Churn prediction completed.\n")
