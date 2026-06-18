import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Fetch a single URL
def fetch_url(url, headers, retries=3):
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                logging.info(f"Successfully fetched {url}")
                return {'url': url, 'html': response.text}
            else:
                logging.warning(f"Failed to fetch {url} (Status code: {response.status_code})")
        except requests.RequestException as e:
            logging.error(f"Error fetching {url}: {e}")
        time.sleep(1)  # Backoff between retries
    return None

# Crawl multiple URLs
def crawl(seed_urls, depth=1, delay=1, max_links=100, max_workers=5, save_interval=10):
    visited = set()
    queue = [(url, 0) for url in seed_urls]
    crawled_data = []
    links_crawled = 0

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            while queue and links_crawled < max_links:
                futures = {executor.submit(fetch_url, url, headers): (url, depth) for url, depth in queue}
                queue = []

                for future in as_completed(futures):
                    result = future.result()
                    url, current_depth = futures[future]

                    if result and url not in visited:
                        visited.add(url)
                        crawled_data.append(result)
                        links_crawled += 1

                        # Save data periodically
                        if links_crawled % save_interval == 0:
                            save_to_json(crawled_data, 'crawled_data_checkpoint.json')

                        if current_depth < depth:
                            soup = BeautifulSoup(result['html'], 'html.parser')
                            for link in soup.find_all('a', href=True):
                                full_url = urljoin(url, link['href'])
                                if full_url not in visited and full_url.startswith(('http://', 'https://')):
                                    queue.append((full_url, current_depth + 1))

                    time.sleep(delay)

        logging.info(f"Crawling completed. Total links crawled: {links_crawled}")
        return crawled_data

    except KeyboardInterrupt:
        # Graceful shutdown on Ctrl + C
        logging.info("Crawling interrupted. Saving crawled data so far...")
        save_to_json(crawled_data, 'crawled_data_checkpoint.json')
        logging.info("Data saved successfully before shutdown.")

    return crawled_data

# Save data to JSON
def save_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    logging.info(f"Data saved to {filename}")

# Read seed URLs from a text file
def read_seed_urls(file_path):
    seed_urls = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            seed_urls = [line.strip() for line in file if line.strip()]
        logging.info(f"Successfully read {len(seed_urls)} seed URLs from {file_path}")
    except Exception as e:
        logging.error(f"Error reading seed URLs from {file_path}: {e}")
    return seed_urls

# Main function to run the steps
def main():
    # File containing seed URLs (one URL per line)
    seed_file = 'urlList.txt'

    # Load seed URLs
    seed_urls = read_seed_urls(seed_file)
    if not seed_urls:
        logging.error("No seed URLs found. Exiting.")
        return

    # Crawl web pages and periodically save crawled data
    crawled_data = crawl(seed_urls, depth=1, max_links=5000, max_workers=5, save_interval=10)

    # After crawling, save all the crawled data
    save_to_json(crawled_data, 'final_crawled_data.json')

if __name__ == "__main__":
    main()
