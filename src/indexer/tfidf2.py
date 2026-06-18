import re
import numpy as np
import pandas as pd
from collections import Counter

# Tokenization
def tokenize(doc):

    rule = r'[\s\~\\!\@\#\$\%\^\&\*\(\)\-\_\+\=\{\}\[\]\;\:\'\"\,\<\.\>\/\?\\|]+'
    re.compile(rule)
    
    terms_ = re.split(rule, doc.lower())
    terms = [term for term in terms_ if term]  # Remove empty terms
    return terms

# Processing function
def process_text(text):

    terms = tokenize(text)
    term_freq = Counter(terms)
    return dict(term_freq)

# Generalized function for TF-IDF computation
def compute_tfidf(csv_path):

    # Load CSV
    try:
        data = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File not found - {csv_path}")
        return None
    
    if 'text' not in data.columns:
        print("Error: The CSV file must have a 'text' column.")
        return None

    # Extract the contents in "texts"
    texts = data['text'].astype(str).tolist()

    # Compute term frequencies for all documents
    term_frequencies = [process_text(text) for text in texts]

    # Create a vocabulary from all documents
    vocab = set()
    for tf in term_frequencies:
        vocab.update(tf.keys())
    vocab = sorted(vocab)  # Sort vocab for consistent ordering
    print(f"Vocabulary size: {len(vocab)}")

    # Initialize a TF matrix
    tfidf_mat = pd.DataFrame(np.zeros((len(vocab), len(texts))), index=vocab)

    # Populate TF matrix
    for doc_idx, tf in enumerate(term_frequencies):
        for term, freq in tf.items():
            tfidf_mat.loc[term, doc_idx] = freq

    # Compute document frequencies
    doc_freq = np.count_nonzero(tfidf_mat, axis=1)

    # Compute IDF
    idf = np.log1p(len(texts) / (1 + doc_freq))  # Improved numerical stability

    # Apply IDF to the TF matrix to get TF-IDF
    tfidf_mat = tfidf_mat.multiply(idf, axis=0)

    # Save the TF-IDF matrix with the index (terms) to a CSV file
    tfidf_mat.to_csv("tfidf_matrix.csv", index=True)

    # Save the IDF values to a CSV file
    idf_df = pd.DataFrame(idf, index=vocab, columns=["IDF"])
    idf_df.to_csv("idf_values.csv")

    print(f"TF-IDF matrix and IDF values saved")

    return tfidf_mat, idf
