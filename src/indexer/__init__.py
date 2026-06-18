# src/indexer/__init__.py

"""
Indexer Module
Transforms raw string tokens into structural TF-IDF numerical vector spaces.
"""

from .tfidf2 import compute_tfidf

__all__ = [
    "compute_tfidf"
]
