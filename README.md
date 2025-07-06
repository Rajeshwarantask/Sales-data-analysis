# Sales Data Analysis: Multinational & Sectoral Hybrid Approach

![sales_trend](https://github.com/user-attachments/assets/a28ba6b5-cb51-4961-897b-89f5e3c65133)
![random_forest_forecast_plot](https://github.com/user-attachments/assets/767f4bb3-8ef7-4417-bc3d-68ffb157e2c6)
![hybrid_forecast_plot](https://github.com/user-attachments/assets/cf0828ae-aa3f-4efe-bc86-12d06d7f763f)
![linear_regression_forecast](https://github.com/user-attachments/assets/022dd8af-4320-4a82-a9ef-68864d54dfc3)
![sales_forecast_seasonality](https://github.com/user-attachments/assets/025d91d5-2572-49e4-aca3-fee501a47b24)
![xgboost_forecast_plot](https://github.com/user-attachments/assets/82b4ff35-728b-4b39-ac8c-80629859159a)

## Project Overview

This repository implements advanced, scalable analytics and predictive modeling for sales, product, and customer data at both sector and multinational corporation (MNC) levels. It leverages hybrid machine learning approaches—integrating time-series, classification, regression, clustering, and ensemble methods—using big data tools. The system supports both prediction (forecasting sales/churn/product demand) and in-depth analysis (segmentation, promotion effectiveness, price sensitivity).

**Scope:**  
- MNC and sector-level sales/product/customer data analysis  
- End-to-end pipeline: data ingestion, preprocessing, EDA, modeling, optimization, visualization  
- Actionable insights for business strategy, inventory, marketing, and customer retention

---

## Objectives

- Predict sales and product demand across product lines, regions, and customer segments
- Analyze customer behavior and segment customers for targeted marketing/churn reduction
- Optimize promotions and pricing strategies using hybrid ML and data-driven analysis
- Support data-driven decision making for MNCs and sector-specific use cases

---

## Tools & Technologies

- **Apache Spark (PySpark, MLlib):** Distributed big data processing and ML modeling
- **Python (scikit-learn, xgboost, prophet, imblearn):** Advanced model-building, tuning, and evaluation
- **Hadoop HDFS / Amazon S3:** Scalable storage for large datasets
- **Jupyter Notebooks / Databricks:** Interactive development and reporting
- **Matplotlib / Seaborn / Plotly:** Data visualization
- **Pandas, NumPy:** Data wrangling and analysis

---

## Data Sources

- **Customer Data:** Demographics, loyalty programs, churn, segmentation
- **Transactional Data:** Sales, discounts, payment methods, product details
- **Product Information:** Reviews, ratings, returns, stock status
- **Promotional Data:** Types, effectiveness, customer targeting
- **Geographical Data:** Store locations, customer proximity
- **Temporal/Seasonal Data:** Holiday/seasonal impact, time-based features

---

## Key Approaches & Hybrid Models

### 1. Sales Forecasting
- **Time Series Models:** Prophet, Linear Regression, Random Forest, XGBoost
- **Feature Engineering:** Date decomposition (month/year), rolling windows
- **Sector/MNC Trends:** Aggregated and segmented forecasts

### 2. Product Demand Analysis
- **Classification:** Decision Trees, Logistic Regression, Random Forest, SVM, Gradient Boosting, XGBoost
- **Regression:** Predicting sales amounts/quantities
- **Hybrid Pipelines:** Use of SMOTE for class imbalance, KNNImputer for missing values
- **Sectoral Focus:** High/low demand product identification by sector or region

### 3. Customer & Churn Analysis
- **Churn Prediction:** Logistic Regression, Random Forest, XGBoost, Gradient Boosting, SVM, and Voting Ensembles
- **Segmentation:** KMeans clustering with PCA visualization, Silhouette scoring
- **Customer Lifetime Value (CLV):** Prioritizing high-value and at-risk customers

### 4. Promotion Optimization
- **Clustering for Targeting:** KMeans on transaction/customer features for segment-specific offers
- **Effectiveness Analysis:** Correlation between discounts, price sensitivity, and sales uplift
- **Dynamic Strategy:** Automated recommendations for promotion types by cluster/segment

### 5. Price Sensitivity & Discount Effectiveness
- **Regression Analysis:** Linear regression of price/discount vs. sales
- **Elasticity Measurement:** Quantifies impact of price changes and discounting on sales

---

## Steps to Implement

1. **Data Preprocessing**
   - Load raw data into Spark; clean/massage with pandas, imputation (KNNImputer), class balancing (SMOTE)

2. **Exploratory Data Analysis (EDA)**
   - Visualize sectoral/MNC trends, outliers, seasonality, and sales/product/customer metrics

3. **Predictive Modeling**
   - Train time series and classification/regression models (with hyperparameter tuning)
   - Evaluate hybrid ensembles for both prediction and explanatory power

4. **Optimization & Deployment**
   - Promotion and inventory optimization using clustering and regression outputs
   - Export best models/results for production use

5. **Visualization & Reporting**
   - Generate interactive dashboards, plots (e.g., sales trends, churn clusters, price sensitivity)

---

## Expected Outcomes

- **MNC & Sector-Level Insights:** Granular forecasts and demand analysis across products, regions, and customer segments
- **Increased Profitability:** High-demand focus, optimized promotions/pricing, reduced stock-outs/waste
- **Enhanced Retention:** Targeted campaigns, churn reduction, improved CLV
- **Actionable Dashboards:** Real-time monitoring and reporting for business leaders

---

## Example: Main Analysis Pipeline

```python
# Load and preprocess data
from data_preprocessing import clean_data
from sales_forecasting import run_forecasting
from churn_prediction import run_churn_prediction
from promotion_optimization import run_promotion_optimization

cleaned_customer_df, cleaned_transaction_df, cleaned_product_df = clean_data(customer_df, transaction_df, product_df)

# Sales forecasting (sector/MNC/product level)
run_forecasting(cleaned_transaction_df)

# Product demand & price sensitivity
from product_demand_analysis import analyze_product_demand
analyze_product_demand(cleaned_transaction_df)

# Customer churn & segmentation
run_churn_prediction(cleaned_customer_df)
from customer_segmentation import run_customer_segmentation
run_customer_segmentation(cleaned_customer_df)

# Promotion targeting/optimization
run_promotion_optimization(cleaned_customer_df, cleaned_transaction_df)
```

---

## Next Steps

- Deploy on Spark/Databricks cluster for large-scale MNC datasets
- Integrate new sector-specific datasets and update feature engineering pipelines
- Expand dashboards and reporting for executive/managerial use

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Apache Spark](https://spark.apache.org/)
- [Python](https://www.python.org/)
- [MLlib](https://spark.apache.org/mllib/)
