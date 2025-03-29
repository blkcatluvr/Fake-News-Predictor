import joblib, os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
from textaugment import EDA
from modelTrainingPipeline.preProcess import clean_text, lemmatize_text, extract_named_entities

#IMPORT MODELS FROM TRAINING
models_dir = "modelTrainingPipeline/models"
misinfo_dir = "modelTrainingPipeline/misInfo"

xgbModel = joblib.load(os.path.join(models_dir, "xgbModel.pkl"))
rfModel = joblib.load(os.path.join(models_dir, "rfModel.pkl"))
sbertModel = SentenceTransformer("modelTrainingPipeline/models/sbertModel")
stackedModel = joblib.load(os.path.join(models_dir, "stackedModel.pkl"))
scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))

tfidf = joblib.load(os.path.join(models_dir, "tfidf.pkl"))
lda = joblib.load(os.path.join(models_dir, "lda.pkl"))
hdbscan = joblib.load(os.path.join(models_dir, "hdbscan.pkl"))
svd = joblib.load(os.path.join(models_dir, "svd.pkl"))
knn = joblib.load(os.path.join(models_dir, "knn.pkl"))
encoder = joblib.load(os.path.join(models_dir, "encoder.pkl"))
selector = joblib.load(os.path.join(models_dir, "selector.pkl"))

misInfoTopics = joblib.load(os.path.join(misinfo_dir, "misInfoTopics.pkl"))
misInfoClusters = joblib.load(os.path.join(misinfo_dir, "misInfoClusters.pkl"))
misInfoVectors = joblib.load(os.path.join(misinfo_dir, "misInfoVectors.pkl"))

eda = EDA()


def preProcessText(text):
    text = clean_text(text)
    return lemmatize_text(text)

def isMisformation(article):
        text = preProcessText(article['content']) #Preprocess Text
        articleEmbedding = sbertModel.encode(text).reshape(1, -1) #Create article embeddings using Sbert to capture semantic meaning
        entities = extract_named_entities(text) #Extracting Named Entites of the Articles
        
        article['text'] = text
        article['articleEmbeddings'] = articleEmbedding
        article['entities'] = entities
        
        #Transforms the text into a numerical vector where each number represents 
        #How important a word is in the document Relative to its occurrence across
        #Articles in the training set
        tfidfX = tfidf.transform([text])
        xTopics = lda.transform(tfidfX)  # Get topic probabilities

        article['topic'] = xTopics.argmax(axis=1)[0] #Assigns to topic with the highest probability
        article['cluster'] = int(knn.predict(tfidfX.toarray())[0]) #Assigns cluster based on TF-IDF features

        #Get a sense of how much of the article is focused on people, organizations, or places.
        article['numPerson'] = (len(entities.get("PERSON", [])))
        article['numOrg'] = (len(entities.get("ORG", [])))
        article['numGpe'] = (len(entities.get("GPE", [])))
        personRatio = article['numOrg']/(len(text.split()) + 1)
        orgRatio = article['numOrg']/(len(text.split()) + 1)
        gpeRatio =  article['numGpe']/(len(text.split()) + 1)
        article['personRatio'] = personRatio
        article['orgRatio'] = orgRatio
        article['gpeRatio'] = gpeRatio

        #Encoding the topic and cluster labels and combines into single vector for predictions
        topicEncoded = encoder.transform([[article['topic']]])
        clusterEncoded = encoder.transform([[article['cluster']]])
        topicClusterTrain = np.hstack((topicEncoded, clusterEncoded))

        #Normalize Ratios to a combined NER feature to get a semantic sense of document
        nerFeats = scaler.transform([[personRatio, orgRatio, gpeRatio]])  
        
        #Combine Meaning of Article via Article Embeddings, NER Features, and Topic/Cluster then Normalize it
        X = np.hstack((articleEmbedding, nerFeats,topicClusterTrain))
        X = selector.transform(X)

        #Comparing the Semantic Vectors of the Article Embeddings and the Known Vectors that are misinformation
        #Then takes the article it is most similar to and creates its novelty 
        similarity = cosine_similarity(articleEmbedding, misInfoVectors)
        novelty = 1 - np.max(similarity)
        stacked_pred = int(stackedModel.predict(X)[0])
        
        #If Predicted Zero and Backed By Novelty it labels it Zero
        #Else if Not Similar to Vectors Just Goes with Predition
        if novelty < 0.05 and stacked_pred==0:
            print("Too similar to misinformation vectors")
            article['label'] = 0
        else:
            article['label'] = stacked_pred

        #Creates augmented Text In the case of Being Outweighed By Real Articles
        article['augmented_text']  = eda.synonym_replacement(text, n=2)

        return article