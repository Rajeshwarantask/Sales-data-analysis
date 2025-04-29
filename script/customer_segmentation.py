import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import functions as F
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from pyspark.sql import functions as F

def run_customer_segmentation(customer_data):
    # Ensure the results directory exists
    os.makedirs("results", exist_ok=True)

    # Drop missing values but preserve 'quantity' and 'customer_id' if available
    required_columns = ['age', 'income_bracket', 'membership_years', 'app_usage',
                        'website_visits', 'social_media_engagement', 'days_since_last_purchase']
    
    customer_data = customer_data.dropna(subset=required_columns)

    # Feature selection
    X = customer_data.select(*required_columns)

    # Encoding for categorical variables
    def map_income(value):
        return F.when(value == 'Low', 1).when(value == 'Medium', 2).when(value == 'High', 3).otherwise(0)

    def map_engagement(value):
        return F.when(value == 'Low', 1).when(value == 'Medium', 2).when(value == 'High', 3).otherwise(0)

    if 'income_bracket' in X.columns:
        X = X.withColumn('income_bracket', map_income(X['income_bracket']))

    if 'social_media_engagement' in X.columns:
        X = X.withColumn('social_media_engagement', map_engagement(X['social_media_engagement']))

    # Select only numerical columns for scaling
    numerical_columns = ['age', 'website_visits', 'days_since_last_purchase']
    X_numerical = X.select(*numerical_columns)

    # Convert to Pandas for scaling
    X_numerical_pandas = X_numerical.toPandas()

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numerical_pandas)

    # Apply KMeans Clustering (Optimal number of clusters using Elbow Method)
    distortions = []
    K_range = range(1, 11)  # Check clusters from 1 to 10
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        distortions.append(kmeans.inertia_)

    # Plot Elbow Method to determine optimal clusters
    plt.figure(figsize=(8, 6))
    plt.plot(K_range, distortions, marker='o')
    plt.title('Elbow Method for Optimal Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Distortion (Inertia)')
    plt.tight_layout()
    plt.savefig("results/elbow_method.png")
    plt.close()

    optimal_clusters = 4  # Example: Based on elbow plot, choose 4 clusters

    # Apply KMeans with optimal clusters
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_scaled)

    # Add the cluster labels back to the original data
    customer_data = customer_data.withColumn('Cluster', F.lit(kmeans_labels))

    # Perform PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Convert PCA results and clusters to a Pandas DataFrame
    pca_df = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
    pca_df['Cluster'] = kmeans_labels

    # Visualize the clusters using Seaborn
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=pca_df, x='PCA1', y='PCA2', hue='Cluster', palette='Set1', s=100)
    plt.title('Customer Segmentation (KMeans)')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.savefig("results/customer_segmentation_pca.png")
    plt.close()

    # Evaluate clustering with silhouette score
    sil_score = silhouette_score(X_scaled, kmeans_labels)
    print(f"Silhouette Score: {sil_score:.2f}")

    # Cluster summary using valid aggregation functions
    cluster_summary = customer_data.groupBy('Cluster').agg(
        F.avg('age').alias('avg_age'),
        F.first('income_bracket').alias('income_bracket_mode'),  # Use first as a representative value
        F.avg('membership_years').alias('avg_membership_years'),
        F.first('app_usage').alias('app_usage_mode'),
        F.avg('website_visits').alias('avg_website_visits'),
        F.first('social_media_engagement').alias('social_media_engagement_mode'),
        F.avg('days_since_last_purchase').alias('avg_days_since_last_purchase')
    )

    cluster_summary.show()

    # Save results
    customer_data.toPandas().to_csv('results/customer_segmentation_results.csv', index=False)
    print("Customer segmentation completed and results saved.")
