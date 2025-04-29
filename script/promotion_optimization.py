import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_promotion_optimization(cleaned_customer_df, cleaned_transaction_df):
    print("Running advanced promotion optimization...")

    if not os.path.exists("results"):
        os.makedirs("results")

    # Convert Spark DataFrame to Pandas
    product_sales_df = cleaned_transaction_df.toPandas()

    # Group by product_id to get aggregated metrics
    product_metrics = product_sales_df.groupby('product_id').agg({
        'total_sales': 'sum',
        'quantity': 'sum',
        'discount_applied': 'mean',
        'avg_items_per_transaction': 'mean'
    }).reset_index()

    product_metrics.columns = ['product_id', 'total_sales', 'total_quantity', 'avg_discount', 'avg_items_per_txn']
    product_metrics.dropna(inplace=True)

    # Standardize the features
    features = product_metrics[['total_sales', 'total_quantity', 'avg_discount', 'avg_items_per_txn']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    product_metrics['Cluster'] = kmeans.fit_predict(scaled_features)

    # Label clusters based on sales
    cluster_perf = product_metrics.groupby('Cluster')['total_sales'].mean().sort_values()
    low_perf, mid_perf, high_perf = cluster_perf.index

    performance_map = {
        low_perf: 'Low Performer',
        mid_perf: 'Mid Performer',
        high_perf: 'Top Performer'
    }
    product_metrics['Performance_Level'] = product_metrics['Cluster'].map(performance_map)

    # Save cluster summary
    summary_stats = product_metrics.groupby('Performance_Level').agg({
        'total_sales': 'mean',
        'total_quantity': 'mean',
        'avg_discount': 'mean',
        'avg_items_per_txn': 'mean',
        'product_id': 'count'
    }).rename(columns={'product_id': 'Product Count'})
    summary_stats.to_csv("results/cluster_summary_stats.csv")

    # Cluster distribution plot
    plt.figure(figsize=(6, 4))
    sns.countplot(data=product_metrics, x='Performance_Level', hue='Performance_Level', palette='Set2', legend=False)
    plt.title("Product Count per Performance Level")
    plt.xlabel("Performance Level")
    plt.ylabel("Number of Products")
    plt.tight_layout()
    plt.savefig("results/cluster_distribution.png")
    plt.close()

    # Scatterplot of sales vs quantity
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=product_metrics, x='total_quantity', y='total_sales', hue='Performance_Level', palette='viridis')
    plt.title("Product Performance Clusters")
    plt.xlabel("Total Quantity Sold")
    plt.ylabel("Total Sales")
    plt.tight_layout()
    plt.savefig("results/product_clusters_advanced.png")
    plt.close()

    # Discount vs Sales by Performance Level
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=product_metrics, x='avg_discount', y='total_sales', hue='Performance_Level', palette='coolwarm')
    plt.title("Discount vs Sales by Performance")
    plt.xlabel("Average Discount")
    plt.ylabel("Total Sales")
    plt.tight_layout()
    plt.savefig("results/discount_vs_sales_by_cluster.png")
    plt.close()

    # Correlation: Discount vs Sales per cluster
    corr_results = {}
    for level in product_metrics['Performance_Level'].unique():
        subset = product_metrics[product_metrics['Performance_Level'] == level]
        corr = subset['avg_discount'].corr(subset['total_sales'])
        corr_results[level] = round(corr, 3)

    correlation_df = pd.DataFrame.from_dict(corr_results, orient='index', columns=['Discount-Sales Correlation'])
    correlation_df.to_csv("results/discount_sales_correlation_by_cluster.csv")

    # Avg items per txn vs sales by cluster
    plt.figure(figsize=(8, 5))
    sns.barplot(data=product_metrics, x='Performance_Level', y='avg_items_per_txn', hue='Performance_Level', palette='pastel', legend=False)
    plt.title("Average Items per Transaction by Cluster")
    plt.xlabel("Performance Level")
    plt.ylabel("Avg Items per Transaction")
    plt.tight_layout()
    plt.savefig("results/avg_items_txn_by_cluster.png")
    plt.close()

    # Promotion strategy insights
    top_products = product_metrics[product_metrics['Performance_Level'] == 'Top Performer']
    mid_products = product_metrics[product_metrics['Performance_Level'] == 'Mid Performer']

    insights = {
        "Top Performer Avg Discount": top_products['avg_discount'].mean(),
        "Mid Performer Avg Discount": mid_products['avg_discount'].mean(),
        "Top Performer Avg Items/Txn": top_products['avg_items_per_txn'].mean(),
        "Mid Performer Avg Items/Txn": mid_products['avg_items_per_txn'].mean(),
        "Top Performer Avg Sales": top_products['total_sales'].mean(),
        "Mid Performer Avg Sales": mid_products['total_sales'].mean(),
    }

    # Strategy recommendation
    recommendations = [
        "📌 Increase promotion for Mid Performers with high discount response.",
        "📌 Reduce discount for Top Performers if sales are strong without high discounts.",
        "📌 Reevaluate Low Performers with high discount but low returns."
    ]

    # Save all insights
    pd.DataFrame([insights]).to_csv("results/promotion_strategy_insights.csv", index=False)
    pd.DataFrame(recommendations, columns=["Recommendation"]).to_csv("results/promotion_recommendations.csv", index=False)

    print("Promotion optimization (advanced) completed.\nInsights and recommendations saved.")
