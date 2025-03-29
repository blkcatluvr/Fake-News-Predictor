import numpy as np
import shap, joblib

from hdbscan import HDBSCAN
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer,InputExample, losses
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
import featureExtraction,modelTraining,preProcess

from torch.utils.data import DataLoader

def main():
    #PreProcess
    df = preProcess.loadData("News_Datasets/DataSet_Misinfo_TRUE.csv", "News_Datasets/DataSet_Misinfo_FAKE.csv")
    df = preProcess.preProcessText(df)
    
    #Feature Extraction
    df = featureExtraction.entityExtraction(df)
    
    # After entity extraction extracts text of real and fake
    real_news = df[df['label'] == 1]['text'].tolist()
    fake_news = df[df['label'] == 0]['text'].tolist()
    train_examples = []

    #Use InputExample to create sentences with Real News Pairs, Fake News Pairs, and Dissimilar Pairing
    for i in range(min(len(real_news), len(fake_news))):
        train_examples.append(InputExample(texts=[real_news[i], real_news[(i+1)%len(real_news)]], label=1.0))
        train_examples.append(InputExample(texts=[fake_news[i], fake_news[(i+1)%len(fake_news)]], label=1.0))
        train_examples.append(InputExample(texts=[real_news[i], fake_news[i]], label=0.0))

    #Trains SBERT to distinguish real and fake news by learning which texts are similar/dissimilar
    sbertModel = SentenceTransformer('all-MiniLM-L6-v2')
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    #Using CosineSimilarityLoss to push similar to have high cosine similarity and dissimlar to have low cosine simiality
    train_loss = losses.CosineSimilarityLoss(sbertModel)
    #Fits model & Assigns Article Embeddings To Text For each Article
    sbertModel.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)
    df['articleEmbeddings'] = df['text'].apply(lambda x: sbertModel.encode(x))
    sbertModel.save("models/sbertModel")

    # Standardize entity ratio features before model training
    df = df.dropna(subset=['text'])
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1,2))
    tfidfX = tfidf.fit_transform(df['text'])
    joblib.dump(tfidf, "models/tfidf.pkl")

    #Train LDA on TF-IDF (Numerical Vectors of Articles & Word Importance)
    lda = LatentDirichletAllocation(n_components=10, random_state=42)
    xTopics = lda.fit_transform(tfidfX)
    joblib.dump(lda, "models/lda.pkl")
    df['topic'] = xTopics.argmax(axis=1)

    #SVD helps compress TF-IDF to then preserve the most important semantic info
    svd = TruncatedSVD(n_components=500)
    tfidfX_reduced = svd.fit_transform(tfidfX)
    joblib.dump(svd, "models/svd.pkl")

    # Train HDBSCAN to Create Clusters from TF-IDF
    hdbscan = HDBSCAN(min_cluster_size=5, metric='euclidean')
    df['cluster'] = hdbscan.fit_predict(tfidfX_reduced)
    joblib.dump(hdbscan, "models/hdbscan.pkl")
    print("New HDBSCAN model trained and saved.")

    #Do the same with KNN because we cannot use predict for HDBSCAN in real time predictions
    knn = KNeighborsClassifier(n_neighbors=5).fit(tfidfX.toarray(), df['cluster'])
    joblib.dump(knn, "models/knn.pkl")
    print("New KNN model trained and saved.")

    #Save Misinformation Topics & Clusters
    misInfoTopics = set(df[df['label'] == 0]['topic'].values)
    misInfoClusters = set(df[df['label'] == 0]['cluster'].values)

    joblib.dump(misInfoTopics, "misInfo/misInfoTopics.pkl")
    joblib.dump(misInfoClusters, "misInfo/misInfoClusters.pkl")

    
    #Model Training
    #Train Test Split 
    dfTrain, dfTest = train_test_split(df, test_size=0.2, random_state=42)
    dfTrain = dfTrain.dropna(subset=['articleEmbeddings'])  # Remove problematic rows
    dfTest = dfTest.dropna(subset=['articleEmbeddings'])

    #Fit Scaler only on Training Data to Prevent Data Leakage
    scaler = StandardScaler()
    scaler.fit(dfTrain[['personRatio', 'orgRatio', 'gpeRatio']])
    joblib.dump(scaler, "models/scaler.pkl")

    #Normalize Ratios to a combined NER feature to get a semantic sense of documents 
    ner_features_train = scaler.transform(dfTrain[['personRatio', 'orgRatio', 'gpeRatio']])
    ner_features_test = scaler.transform(dfTest[['personRatio', 'orgRatio', 'gpeRatio']])
    
    #Converting Embeddings into 2D Array for Training Later
    article_embeddings_train = np.vstack(dfTrain['articleEmbeddings'].values)
    article_embeddings_test = np.vstack(dfTest['articleEmbeddings'].values)

    #Encoding the topic and cluster labels and combines into single vector for predictions
    #Has to do for both testing and training sets
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    topicTrain = dfTrain[['topic']].values
    topicTest = dfTest[['topic']].values
    clusterTrain = dfTrain[['cluster']].values
    clusterTest = dfTest[['cluster']].values
    encoder.fit(np.vstack((topicTrain, clusterTrain)))
    topicTrainEncoded = encoder.transform(topicTrain)
    clusterTrainEncoded = encoder.transform(clusterTrain)
    topicTestEncoded = encoder.transform(topicTest)
    clusterTestEncoded = encoder.transform(clusterTest)
    topicClusterTrain = np.hstack((topicTrainEncoded, clusterTrainEncoded))
    topicClusterTest = np.hstack((topicTestEncoded, clusterTestEncoded))
    joblib.dump(encoder, "models/encoder.pkl")
    
    # Stack SBERT sentence embeddings with NER features
    X_train = np.hstack((article_embeddings_train, ner_features_train, topicClusterTrain))
    X_test = np.hstack((article_embeddings_test, ner_features_test, topicClusterTest))

    selector = SelectKBest(score_func=f_classif, k=300)  # Choose top 300 features
    y_train = dfTrain['label'].values #Predicting Labels Fake or Real News
    y_test = dfTest['label'].values
    X_train = selector.fit_transform(X_train, y_train)
    X_test = selector.transform(X_test)
    joblib.dump(selector, "models/selector.pkl")

    #Saves Article Embeddings From Fake News Articles to Compare with for later 
    misInfoVectors = np.vstack(dfTrain[dfTrain['label'] == 0]['articleEmbeddings'].values)
    joblib.dump(misInfoVectors,"misInfo/misInfoVectors.pkl")
    df.to_csv("News_Datasets/trained.csv", index=False)


    #Intializing and Fitting Models
    xgbModel, rfModel = modelTraining.trainModels(X_train,y_train)

    #Stacking Modls
    stackedModel = StackingClassifier(
        estimators=[
            ('xgb', xgbModel),
            ('rf', rfModel)
        ],
        final_estimator=LogisticRegression()
    )

    # Trains and validates data 5 times to output F1 score which is often better for imbalanced datasets
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(stackedModel, X_train, y_train, cv=cv, scoring='f1')
    print("Cross-validated F1 Score:", scores.mean())

    #Fit Stack Model and Predict X Test 
    stackedModel.fit(X_train, y_train)
    predictions = stackedModel.predict(X_test)

    #Save Model and DataSet
    joblib.dump(stackedModel, "models/stackedModel.pkl")
    df.to_csv("News_Datasets/trained.csv", index=False)

    #Performance and Accuracy
    aucScore = roc_auc_score(y_test, predictions)
    print("AUC-ROC Score:", aucScore)
    print(classification_report(y_test, predictions))

    if isinstance(stackedModel.final_estimator_, LogisticRegression):
        explainer = shap.Explainer(stackedModel.predict, X_train, max_evals=700)
        shap_values = explainer(X_test)
        shap.summary_plot(shap_values, X_test, show=False)
        plt.savefig("shap_summary_plot.png", bbox_inches='tight')
        plt.close()
    else:
        print("SHAP not supported for this model type.")
if __name__ == "__main__":
    main()