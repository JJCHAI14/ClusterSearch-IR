import pandas as pd
import numpy as np


def kmeans_from_scratch(tf_idf, k, max_iterations=300, tol=1e-4):

    # Step 1: Initialize centroids
    centroids = initialize_centroids(tf_idf, k)

    for iteration in range(max_iterations):
        # Step 2: Assign clusters
        labels = assign_clusters(tf_idf, centroids)

        # Step 3: Update centroids
        new_centroids = update_centroids(tf_idf, labels, k)

        # Check for convergence
        if np.linalg.norm(new_centroids - centroids) < tol:
            print(f"Converged in {iteration + 1} iterations.")
            break

        centroids = new_centroids

    return {
        "labels": labels,
        "centroids": centroids,
        "iterations": iteration + 1
    }

def initialize_centroids(tf_idf, k):

    n_samples = tf_idf.shape[0]
    random_indices = np.random.choice(n_samples, k, replace=False)

    centroids = tf_idf.iloc[random_indices].to_numpy()  # Convert selected rows to numpy array
    return centroids

def assign_clusters(tf_idf, centroids):

    tf_idf_matrix = tf_idf.to_numpy()  # Convert to numpy array for computation

    # Compute cosine similarity
    # Note: np.linalg.norm function -> calculate the Euclidean norm (distance) of a vector
    dot_products = np.dot(tf_idf_matrix, centroids.T)
    centroid_norms = np.linalg.norm(centroids, axis=1)
    tf_idf_norms = np.linalg.norm(tf_idf_matrix, axis=1).reshape(-1, 1)

    cosine_similarity = dot_products / (tf_idf_norms * centroid_norms)

    # Assign clusters based on the highest similarity
    labels = np.argmax(cosine_similarity, axis=1)
    return labels

def update_centroids(tf_idf, labels, k):

    tf_idf_matrix = tf_idf.to_numpy()  # Convert to numpy array for computation
    new_centroids = []

    for cluster in range(k):
        # Get all data points assigned to the current cluster
        cluster_points = tf_idf_matrix[labels == cluster]

        if len(cluster_points) > 0:
            # Compute the mean of the cluster points
            centroid = np.mean(cluster_points, axis=0)
        else:
            # Handle empty clusters by reinitializing randomly
            centroid = tf_idf_matrix[np.random.choice(tf_idf_matrix.shape[0])]

        new_centroids.append(centroid)

    return np.array(new_centroids)

