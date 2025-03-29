from kafka import KafkaConsumer
from pymongo import MongoClient
import json, isFake
import numpy as np

#Function to make types compatible with MongoDB
def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(x) for x in obj]
    return obj


password = "YOUR MONGODB PASSWORD"  # or whatever your actual password is
MONGO_URI = f"YOUR MONGODB URI"
client = MongoClient(MONGO_URI)
db = client["newsDatabase"]
collection = db["articles"]

fakeCount = collection.count_documents({'label': 0})
realCount = collection.count_documents({'label': 1})

# Kafka Consumer Setup
consumer = KafkaConsumer(
    'newsData',  # Kafka topic name
    bootstrap_servers=['localhost:9092'], 
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))  # Deserialize JSON
)

#Now Waiting For Articles Received From Kafka & Appends to Consumer
print("Listening for messages on newsData...")


for message in consumer:
    news_item = message.value

    #Skip if Title Already Exists in MongoDB Collection 
    if collection.find_one({'title': news_item['title']}):
        print(f"Skipping duplicate article: {news_item['title']}")
        continue
    
    print(f"Received: {news_item}")
    
    #Determines Label (Fake News: 0, Real News: 1)
    #Creates a label for the Real Article 
    article = isFake.isMisformation(news_item)
    misinformation = isFake.isMisformation({**article, 'text': article['augmented_text']})
    #Creates An Article that uses Augmented Text to create a fake label
    max_attempts = 5
    attempt = 0
    while(misinformation['label'] == 1 and attempt < max_attempts):
        misinformation = isFake.isMisformation({**misinformation, 'text': misinformation['augmented_text']})
        attempt += 1
    
    if article['label'] == 0:
        print("Article is likely to contain misinformation")
    else:
        print("Article is likely to contain accurate information")
    
    #Pops unique iD to avoid duplicate id errors if need to update item
    news_item.pop('_id', None)
    misinformation.pop('_id', None)

    # Want to keep even amount of real and fake articles in database
    # Adds the article of lesser count in database
    if fakeCount < realCount:
        collection.update_one({'title': misinformation['title']},{'$set': convert_numpy(misinformation)},upsert=True)
    elif realCount < fakeCount:
        collection.update_one({'title': news_item['title']},{'$set': convert_numpy(news_item)},upsert=True)
    else:
        collection.update_one({'title': news_item['title']},{'$set': convert_numpy(news_item)},upsert=True)
        collection.update_one({'title': misinformation['title']},{'$set': convert_numpy(misinformation)},upsert=True)

    print("Saved to MongoDB Atlas")
