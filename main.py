import os
import pandas as pd
import json
import logging
from dotenv import load_dotenv
# Core Engine Architectural Components
from src.crawler import read_seed_urls, crawl, save_to_json
from src.indexer import compute_tfidf
from src.clustering import (
    kmeans_from_scratch, 
    analyze_cluster_terms, 
    evaluate_clustering, 
    visualize_clusters
)
from src.utils import process_crawled_data, save_to_csv, save_clusters_to_csv

load_dotenv()

max_depth = int(os.getenv("MAX_CRAWL_DEPTH", 1))
max_links = int(os.getenv("MAX_CRAWL_LINKS", 5000))
max_workers = int(os.getenv("MAX_CRAWL_WORKERS", 5))


# ======= STEP 1: Crawl website from given seed url
seed_file = 'urlList.txt'  # Text file containing seed URLs

# Load seed URLs
seed_urls = read_seed_urls(seed_file)
if not seed_urls:
    logging.error("No seed URLs found. Exiting.")

# Crawl web pages
crawled_data = crawl(seed_urls, depth=max_depth, max_links=max_links, max_workers=max_workers)
save_to_json(crawled_data, 'crawled_data.json')

# ======= STEP 2: Process the saved url and its content =======
try:
    # Load crawled data
    with open('crawled_data.json', 'r', encoding='utf-8') as f:
        crawled_data = json.load(f)
    logging.info("Successfully loaded crawled_data.json.")

    # Process the data
    processed_data = process_crawled_data(crawled_data)

    # Save processed data to CSV
    save_to_csv(processed_data, 'cleaned_data.csv')

except Exception as e:
    logging.error(f"An error occurred: {e}")


    
# ======= STEP 3: Compute tf-idf matrix from given documents =======
csv_path = "cleaned_data.csv"  # Replace with your CSV file path
tfidf_matrix, _ = compute_tfidf(csv_path)
print("\nTF-IDF Matrix:")
print(tfidf_matrix) # Output format: row refer to terms, column refer to the docs [!!! Important !!!]


# ======= STEP 4: Perform Clustering on tf-idf =======

num_clusters = int(os.getenv("NUM_CLUSTERS", 13))
tolerance = float(os.getenv("CLUS_TOLERANCE", 1e-4))
max_iteration = int(os.getenv("CLUS_MAX_ITERATION", 300))

# Transpose to: row refer to docs, column refer to the terms [!!! Important !!!]
tfidf_matrix_transposed = tfidf_matrix.T  # Ensure that each row is a document vector

results = kmeans_from_scratch(tfidf_matrix_transposed, num_clusters)
print("Kmeans:")
print("Cluster Labels:", results["labels"])
print("Centroids:", results["centroids"])
print("")


# ======= STEP 5: Analyze Clusters =======
# Extract centroids from the results
centroids = results['centroids']

print("Kmeans:")
# Perform cluster analysis
top_terms_per_cluster = analyze_cluster_terms(tfidf_matrix, results['centroids'], top_n=3)

# Output the top terms for each cluster in the desired format
print("Top Terms per Cluster:")
for cluster_index, top_terms in top_terms_per_cluster.items():
    print(f"Cluster {cluster_index}: {', '.join(top_terms)}")



# ======= STEP 6: Evaluate Clustering Performance =======

# Extract label from the results
evaluation_metrics = evaluate_clustering(tfidf_matrix_transposed, results['labels'], results['centroids'])

print(f"Silhouette Score: {evaluation_metrics['silhouette_score']}")
print(f"Inertia: {evaluation_metrics['inertia']}")


#  ======= STEP 7: Visualize Clustering Results =======
visualize_clusters(tfidf_matrix_transposed, results['labels'], results['centroids'])


# ====== STEP 8: Save the url of each cluster and their top key terms into csv file seperately ======

urls = pd.read_csv(csv_path)["url"].tolist()  # Assuming your CSV file has a "url" column
dir_path = 'kmeansResults'

# Save clusters to CSV files
save_clusters_to_csv(results['labels'], urls, top_terms_per_cluster, results['centroids'], dir_path)



