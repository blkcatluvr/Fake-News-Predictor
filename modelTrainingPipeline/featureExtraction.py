from sklearn.metrics.pairwise import cosine_similarity

def get_novelty_score(text, misInfoVectors, sbert_model):
    vector = sbert_model.encode(text).reshape(1, -1)  # Convert new article to vector
    similarity = cosine_similarity(vector, misInfoVectors)
    return 1 - max(similarity)  # Higher score = more novel
def entityExtraction(df):
    # Compute raw counts of named entities
    df['numPerson'] = df['entities'].apply(lambda x: len(x.get("PERSON", [])))
    df['numOrg'] = df['entities'].apply(lambda x: len(x.get("ORG", [])))
    df['numGpe'] = df['entities'].apply(lambda x: len(x.get("GPE", [])))
    # Normalize entity counts by text length (+1 to prevent division by zero)
    df['personRatio'] = df['numPerson'] / (df['text'].apply(lambda x: len(x.split())) + 1)
    df['orgRatio'] = df['numOrg'] / (df['text'].apply(lambda x: len(x.split())) + 1)
    df['gpeRatio'] = df['numGpe'] / (df['text'].apply(lambda x: len(x.split())) + 1)
    return df