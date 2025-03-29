import spacy,re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict

stop_words = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_sm')

def loadData(pathT,pathF):
    dfTrue = pd.read_csv(pathT)
    dfFake = pd.read_csv(pathF)
    dfTrue['label'] = 1
    dfFake['label'] = 0
    df = pd.concat([dfTrue.sample(n=10000, random_state=42), dfFake.sample(n=10000, random_state=42)]).sample(frac=1, random_state=42)  # Shuffle dataset
    return df

def preProcessText(df):
    df['text'] = df['text'].apply(clean_text)
    df['entities'] = df['text'].apply(extract_named_entities)
    df['text'] = df['text'].apply(lemmatize_text)
    return df

def clean_text(text):
    if not isinstance(text, str):
      return ""
    text = re.sub(r"[^\w\s.!?]", "", text)
    text = text.lower()  # Convert to lowercase
    tokens = word_tokenize(text)  # Tokenization
    return " ".join(tokens)

def lemmatize_text(text):
    doc = nlp(text)
    lemmatized_tokens = [token.lemma_ for token in doc]
    return " ".join(lemmatized_tokens)

def extract_named_entities(text):
    doc = nlp(text)
    entities = defaultdict(list)
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE"]:
            entities[ent.label_].append(ent.text)
    return entities
