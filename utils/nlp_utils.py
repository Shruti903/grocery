import re
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download resources silently
def _download_nltk_resources():
    resources = ['averaged_perceptron_tagger', 'wordnet', 'omw-1.4']
    for res in resources:
        try:
            nltk.data.find(f'taggers/{res}' if 'tagger' in res else f'corpora/{res}')
        except LookupError:
            nltk.download(res, quiet=True)

_download_nltk_resources()

CONTEXT_MAPPING = {
    "pasta": ["pasta", "tomato", "cheese", "olive oil", "sauce", "vegetables"],
    "breakfast": ["milk", "bread", "eggs", "butter", "cereals", "pancake"],
    "juice": ["fruits", "juice", "citrus"],
    "healthy": ["salad", "chicken", "broccoli", "apples", "yogurt"]
}

def get_context_items(text):
    """
    Scans the input for mapped intent triggers (e.g., 'pasta').
    Returns mapped ingredients to prevent random noun scraping.
    """
    text_lower = text.lower()
    mapped_items = []
    detected_contexts = []
    
    for intent, items in CONTEXT_MAPPING.items():
        if intent in text_lower:
            mapped_items.extend(items)
            detected_contexts.append(intent)
            
    return list(set(mapped_items)), detected_contexts

def extract_entities(text):
    """
    Lightweight Named Entity Recognition to extract prospective food items.
    Focuses on Nouns (NN, NNS) as food usually falls in this category.
    """
    tokens = word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    
    entities = []
    current_entity = []
    
    # We want to extract single or compound nouns (e.g. "fruit juice")
    for word, tag in pos_tags:
        if word.lower() in {"cook", "make", "want", "breakfast", "dinner", "lunch"}:
            continue # Skip obvious verb-like or event nouns that are not groceries
            
        if tag.startswith('NN'):
            current_entity.append(word.lower())
        else:
            if current_entity:
                entities.append(' '.join(current_entity))
                current_entity = []
                
    if current_entity:
         entities.append(' '.join(current_entity))
         
    return entities

def expand_synonyms(word, dataset_vocabulary=None):
    """
    Use WordNet to expand synonyms for a given word.
    Optionally, filter by words that actually exist in the dataset vocabulary.
    """
    synonyms = set()
    synonyms.add(word)
    
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            # Replace underscores with spaces (WordNet style -> natural text)
            syn_word = lemma.name().replace('_', ' ').lower()
            synonyms.add(syn_word)
            
    # If a dataset vocabulary is provided, we can filter for relevant words
    if dataset_vocabulary is not None:
        # Check if the synonym has a partial match in the dataset vocab
        relevant_synonyms = set()
        for syn in synonyms:
            if len(syn) <= 2:
                continue # Prevent single-letter lemmas (like 'c' for cold) from matching everything
            for vocab_item in dataset_vocabulary:
                if syn in vocab_item or vocab_item in syn:
                    relevant_synonyms.add(vocab_item)
        return list(relevant_synonyms)
        
    return list(synonyms)

def semantic_match(user_items, dataset_items, threshold=0.2):
    """
    Uses TF-IDF and cosine similarity to map user-input items to the closest matching 
    items in the actual groceries dataset. 
    `dataset_items` is a list of unique items from the dataset.
    Returns a dictionary of matched item and its confidence score.
    """
    if len(user_items) == 0 or len(dataset_items) == 0:
        return {}
        
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
    
    # Fit vectorizer on all dataset items + user items to build comprehensive vocabulary
    corpus = list(dataset_items) + user_items
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    dataset_tfidf = tfidf_matrix[:len(dataset_items)]
    user_tfidf = tfidf_matrix[len(dataset_items):]
    
    similarities = cosine_similarity(user_tfidf, dataset_tfidf)
    
    matches = {}
    for i, user_item in enumerate(user_items):
        best_idx = np.argmax(similarities[i])
        best_score = similarities[i][best_idx]
        
        if best_score >= threshold:
            best_match = dataset_items[best_idx]
            # Avoid overwriting with a worse score
            if best_match not in matches or matches[best_match] < best_score:
                matches[best_match] = round(best_score, 2)
                
    return matches
