import requests
from kafka import KafkaProducer
import json
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Kafka Producer Configuration
KAFKA_TOPIC = "newsData"
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"

# Initialize Kafka Producer
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

API_KEY = "GNEWS API KEY"
CATEGORY = "general"
API_URL = f"https://gnews.io/api/v4/top-headlines?category={CATEGORY}&lang=en&country=us&max=10&apikey={API_KEY}"

# Track sent articles to avoid duplicates
seen_articles = set()

def fetch_news():
    """Fetches news from GNews API and sends new articles to Kafka."""
    try:
        response = requests.get(API_URL, timeout=10)  # Timeout to prevent hanging
        response.raise_for_status()  # Raise exception for HTTP errors
        
        articles = response.json().get("articles", [])
        new_articles = 0
        
        for article in articles:
            if article["title"] not in seen_articles:  # Avoid duplicates
                seen_articles.add(article["title"])
                producer.send(KAFKA_TOPIC, article)
                new_articles += 1
                logging.info(f"Sent to Kafka: {article['title']}")

        producer.flush()  # Ensure messages are sent immediately
        logging.info(f"{new_articles} new articles sent to Kafka.")
    
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching news: {e}")
    
    except Exception as e:
        logging.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    try:
        while True:
            fetch_news()
            logging.info("Waiting 15 minutes before fetching again...")
            time.sleep(900)  # Wait 15 minutes before the next request to avoid request limits
    except KeyboardInterrupt:
        logging.info("Script interrupted. Closing Kafka producer.")
    finally:
        producer.close()  # Ensure producer is closed before exit

