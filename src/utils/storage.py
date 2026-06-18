import os
import pandas as pd
import numpy as np

def save_clusters_to_csv(labels, urls, top_terms_per_cluster, centroids, output_dir="dbscanResults"):

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Save Key Terms for Each Cluster
    key_terms_data = []
    for cluster, terms in top_terms_per_cluster.items():
        key_terms_data.append([cluster, ", ".join(terms)])  # Join key terms with commas

    # Save the key terms to a CSV file
    key_terms_df = pd.DataFrame(key_terms_data, columns=["Cluster", "Key Terms"])
    key_terms_df.to_csv(os.path.join(output_dir, "key_terms.csv"), index=False)
    print("Key terms saved to 'key_terms.csv'")

    # 2. Save URLs for Each Cluster with Their Respective Index from `labels`
    cluster_urls = {cluster: [] for cluster in top_terms_per_cluster.keys()}
    for idx, (url, label) in enumerate(zip(urls, labels)):
        cluster_urls[label].append([idx, url])  # Use the index from the labels

    # Write each cluster's URLs to a separate CSV file with index number from labels
    for cluster, urls_in_cluster in cluster_urls.items():
        url_file_path = os.path.join(output_dir, f"cluster_{cluster}_urls.csv")
        url_df = pd.DataFrame(urls_in_cluster, columns=["Index", "URL"])
        url_df.to_csv(url_file_path, index=False)
        print(f"Cluster {cluster} URLs saved to {url_file_path}")

    # 3. Save Centroids for Each Cluster
    centroid_data = []
    for i, centroid in enumerate(centroids):
        centroid_data.append([i, ", ".join([str(c) for c in centroid])])  # Join centroid values with commas

    # Save the centroids to a CSV file
    centroids_df = pd.DataFrame(centroid_data, columns=["Cluster", "Centroid"])
    centroids_df.to_csv(os.path.join(output_dir, "centroids.csv"), index=False)
    print("Centroids saved to 'centroids.csv'")
