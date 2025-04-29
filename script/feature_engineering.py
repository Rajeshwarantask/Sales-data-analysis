import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

def robust_date_conversion(df, date_col='transaction_date'):
    """Convert transaction_date to datetime and extract date-based features."""
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df['year_month'] = df[date_col].dt.to_period('M')
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['month'] = df[date_col].dt.month
    df['quarter'] = df[date_col].dt.quarter
    return df

def add_monthly_sales_trends(df):
    """Add monthly aggregated sales as a trend feature."""
    monthly_sales = df.groupby('year_month')['sales_amount'].sum().reset_index()
    monthly_sales['monthly_sales_trends'] = monthly_sales['sales_amount']
    df = pd.merge(df, monthly_sales[['year_month', 'monthly_sales_trends']], on='year_month', how='left')
    return df

def add_customer_lifetime_value(df):
    """Add customer lifetime value based on sales, frequency, and lifespan."""
    avg_order_value = df.groupby('customer_id')['sales_amount'].mean().reset_index(name='avg_order_value')
    purchase_freq = df.groupby('customer_id')['transaction_id'].count().reset_index(name='purchase_frequency')
    
    lifespan = df.groupby('customer_id')['transaction_date'].agg(['min', 'max']).reset_index()
    lifespan['lifespan_days'] = (pd.to_datetime(lifespan['max']) - pd.to_datetime(lifespan['min'])).dt.days
    lifespan = lifespan[['customer_id', 'lifespan_days']]
    
    df = df.merge(avg_order_value, on='customer_id', how='left')
    df = df.merge(purchase_freq, on='customer_id', how='left')
    df = df.merge(lifespan, on='customer_id', how='left')
    
    df['customer_lifetime_value'] = df['avg_order_value'] * df['purchase_frequency'] * (df['lifespan_days'] / 365)
    return df

def add_days_between_purchases(df):
    """Calculate days between purchases for each customer."""
    df = df.sort_values(by=['customer_id', 'transaction_date'])
    df['days_between_purchases'] = df.groupby('customer_id')['transaction_date'].diff().dt.days
    df['days_between_purchases'] = df['days_between_purchases'].fillna(0)
    return df

def add_average_basket_size(df):
    """Estimate average basket size: quantity × product variety per customer."""
    product_variety = df.groupby('customer_id')['product_category'].transform('nunique')
    df['average_basket_size'] = df['quantity'] * product_variety
    return df

def add_seasonal_trends(df):
    """Incorporate average sales by month as seasonal trend."""
    if 'month' not in df.columns:
        df['month'] = df['transaction_date'].dt.month
    seasonal_avg = df.groupby('month')['sales_amount'].mean().reset_index(name='seasonal_sales_trends')
    df = df.merge(seasonal_avg, on='month', how='left')
    return df

def handle_missing_values(df):
    """Handle missing numerical values using KNN imputation."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    imputer = KNNImputer(n_neighbors=5)
    df_numeric = pd.DataFrame(imputer.fit_transform(df[numeric_cols]), columns=numeric_cols)
    df[numeric_cols] = df_numeric
    return df

def scale_numeric_features(df, exclude_cols=None):
    """Standardize numerical features excluding ID columns."""
    if exclude_cols is None:
        exclude_cols = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.difference(exclude_cols)
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def feature_engineering_pipeline(df):
    """Apply the full pipeline of feature engineering transformations."""
    df = robust_date_conversion(df)
    df = add_monthly_sales_trends(df)
    df = add_customer_lifetime_value(df)
    df = add_days_between_purchases(df)
    df = add_average_basket_size(df)
    df = add_seasonal_trends(df)
    df = handle_missing_values(df)
    df = scale_numeric_features(df, exclude_cols=['customer_id', 'product_id'])
    return df

if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data/retail_data.csv')
    
    # Apply feature engineering
    df = feature_engineering_pipeline(df)
    
    # Save processed dataset
    df.to_csv('data/retail_data_with_features.csv', index=False)
    print("✅ Feature engineering completed and saved.")
