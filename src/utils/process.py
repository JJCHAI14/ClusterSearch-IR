import json
import csv
import logging
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Ensure NLTK resources are downloaded
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Fix deterministic behavior in langdetect
DetectorFactory.seed = 0

# Clean HTML content
def clean_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    for script_or_style in soup(['script', 'style']):
        script_or_style.decompose()
    return ' '.join(soup.stripped_strings)

# Detect language of text
def is_english(text):
    try:
        language = detect(text)
        return language == 'en'
    except LangDetectException:
        return False

# Preprocess text
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Process crawled data
def process_crawled_data(crawled_data):
    processed_data = []
    for entry in crawled_data:
        try:
            raw_text = clean_html(entry['html'])
            if is_english(raw_text):
                cleaned_text = preprocess_text(raw_text)
                processed_data.append({'url': entry['url'], 'text': cleaned_text})
            else:
                logging.info(f"Non-English content detected and skipped for URL: {entry['url']}")
        except Exception as e:
            logging.error(f"Error processing {entry['url']}: {e}")
    return processed_data

# Save data to CSV
def save_to_csv(data, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['url', 'text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in data:
            writer.writerow(entry)
    logging.info(f"Data saved to {filename}")

# Main function
def main():
    try:
        # Load crawled data
        with open('final_crawled_data.json', 'r', encoding='utf-8') as f:
            crawled_data = json.load(f)
        logging.info("Successfully loaded crawled_data.json.")

        # Process the data
        processed_data = process_crawled_data(crawled_data)

        # Save processed data to CSV
        save_to_csv(processed_data, 'cleaned_data.csv')

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
