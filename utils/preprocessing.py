import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Attempt to download required NLTK resources gracefully
def _download_nltk_resources():
    resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'averaged_perceptron_tagger']
    for res in resources:
        try:
            nltk.data.find(f'tokenizers/{res}')
        except LookupError:
            try:
                nltk.data.find(f'corpora/{res}')
            except LookupError:
                try:
                    nltk.data.find(f'taggers/{res}')
                except LookupError:
                    nltk.download(res, quiet=True)

_download_nltk_resources()

def clean_text(text):
    """
    Lowercases the text and removes punctuation.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_and_lemmatize(text):
    """
    Tokenizes text, removes stopwords, and lemmatizes words.
    """
    if not text:
        return []
        
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    # Make sure we don't remove food-related stopwords if any exist, though usually they don't overlap much.
    
    tokens = word_tokenize(text)
    processed_tokens = [
        lemmatizer.lemmatize(word) 
        for word in tokens 
        if word not in stop_words and len(word) > 1
    ]
    
    return processed_tokens

def detect_intent(text):
    """
    Detects user intent based on predefined keyword categories.
    Returns a dictionary of boolean flags.
    """
    intents = {
        'cooking': False,
        'breakfast': False,
        'healthy': False,
        'snacks': False
    }
    
    text_lower = text.lower()
    
    # Keyword sets for intent classification
    cooking_keywords = {'cook', 'make', 'prepare', 'recipe', 'bake', 'dinner', 'lunch', 'meal'}
    breakfast_keywords = {'breakfast', 'morning', 'brunch'}
    healthy_keywords = {'healthy', 'diet', 'organic', 'vegan', 'keto', 'low fat', 'fresh', 'nutritious'}
    snacks_keywords = {'snack', 'party', 'movie', 'sweet', 'junk', 'treat'}
    
    tokens = set(word_tokenize(text_lower))
    
    if any(kw in tokens or kw in text_lower for kw in cooking_keywords):
        intents['cooking'] = True
        
    if any(kw in tokens or kw in text_lower for kw in breakfast_keywords):
        intents['breakfast'] = True
        
    if any(kw in tokens or kw in text_lower for kw in healthy_keywords):
        intents['healthy'] = True
        
    if any(kw in tokens or kw in text_lower for kw in snacks_keywords):
        intents['snacks'] = True
        
    return intents
