Fake News Detection Pipeline

Real-time news classification system that identifies misinformation by leveraging NLP, clustering, and machine learning techniques. Built with Python, Scikit-learn, Kafka, MongoDB Atlas, and SBERT embedding

Overview

This project consumes live news articles, analyzes them using multiple NLP techniques, and classifies them as Fake or Real. The classification results are stored in MongoDB for future model training data. This project encompasses big data and model pipelining.
I initally trained the data using two separate datasets from Kaggle and preprocessing them and taking 10000 real & 10000 fake articles. To turn the content of the articles into useful input for our machine learning models I incorporated text embeddings, vectorization,
clustering, topic modeling, and classification techniques. After my model was trained it scored at classifying real vs fake articles at 97% accuracy. I then took this model and incoprated it into my ML pipeline. From GNews I was pull articles and post them on Kafka.
I would then retrieve articles from Kafka, make my model predict their label of real or fake then post to article to my MongoDB collection. If there are far more real or fake articles I integrated data augmentation to create a fake article for each real article.

Key Features
- Real-Time News Processing using Apache Kafka & GNews API Endpoint
- Advanced Feature Extraction with:
  - TF-IDF Vectorization
  - SBERT Embeddings (Semantic Similarity)
  - Named Entity Recognition (NER)
  - Topic Modeling (LDA)
  - Clustering (HDBSCAN + KNN)
- Stacked ML Model: XGBoost and Random Forest.
- Text Augmentation with Easy Data Augmentation (EDA)
- MongoDB Integration to persist enriched news articles

Architecture:
gatheringNews.py --> Kafka Topic: newsData --> kafkaNewsConsumer.py (Checks MongoDB, Calls isFake.py) --> isFake.py(Loads models, Extracts features, Predicts & Augments) --> MongoDB Atlas 
