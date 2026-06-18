# src/crawler/__init__.py

"""
Crawler Module
Handles concurrent seed extraction, multi-threaded crawling, and raw payload serialization.
"""

from .crawler2 import read_seed_urls, crawl, save_to_json

__all__ = [
    "read_seed_urls",
    "crawl",
    "save_to_json"
]
