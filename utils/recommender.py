import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

NON_FOOD_CATEGORIES = [
    'cleaner', 'cosmetic', 'detergent', 'soap', 'pet care', 
    'photo/film', 'newspapers', 'shopping bags', 'flower', 
    'plants', 'hair spray', 'hygiene', 'candles', 'house keeping products', 
    'female sanitary products', 'decalcifier', 'cling film/bags', 'kitchen utensil'
]

def load_dataset(filepath):
    """
    Loads the grocery dataset and immediately FILTERS OUT non-food related items.
    """
    try:
        if hasattr(filepath, 'seek'):
            filepath.seek(0)
            
        df = pd.read_csv(filepath)
        
        # Filter logic to remove non-edible categories
        if 'itemDescription' in df.columns:
            # We create a boolean mask: True if the item does NOT contain any of the non-food keywords
            mask = ~df['itemDescription'].str.lower().apply(
                lambda x: any(nf in x for nf in NON_FOOD_CATEGORIES) if isinstance(x, str) else False
            )
            df = df[mask]
            
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return pd.DataFrame()

def get_item_frequencies(df):
    """
    Calculates the frequency of each unique item in the dataset.
    Returns a sorted Series of frequencies.
    """
    if df.empty or 'itemDescription' not in df.columns:
        return pd.Series(dtype=int)
        
    freq = df['itemDescription'].value_counts()
    return freq

def build_association_rules(df, min_support=0.001, min_confidence=0.05):
    """
    Prepares the dataset into a basket format and runs Apriori to generate rules.
    This can be expensive, so it should be cached by Streamlit.
    """
    if df.empty or 'itemDescription' not in df.columns or 'Member_number' not in df.columns:
        return pd.DataFrame()
        
    # Group by Member and Date to form baskets of transactions
    basket = (df.groupby(['Member_number', 'Date', 'itemDescription'])['itemDescription']
                .count().unstack().reset_index().fillna(0)
                .set_index(['Member_number', 'Date']))

    # Convert counts to boolean presence (1/0)
    # The new mlxtend apriori expects boolean values
    def encode_units(x):
        if x <= 0:
            return False
        if x >= 1:
            return True
            
    basket_sets = basket.map(encode_units)
    
    # Generate frequent itemsets
    frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)
    
    # Generate association rules
    if frequent_itemsets.empty:
        return pd.DataFrame()
        
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    
    # Sort rules by lift to prioritize strong associations
    rules = rules.sort_values('lift', ascending=False)
    
    return rules

def recommend_items(user_items, rules, top_n=5):
    """
    Recommends items based on the Apriori rules and the current user items.
    """
    if rules is None or rules.empty or not user_items:
        return []
        
    recommendations = set()
    user_items_set = set(user_items)
    
    for idx, rule in rules.iterrows():
        # Check if the user has any of the antecedents
        antecedents = set(rule['antecedents'])
        consequents = set(rule['consequents'])
        
        if antecedents.issubset(user_items_set):
            # Only recommend items the user hasn't already listed
            new_items = consequents - user_items_set
            recommendations.update(new_items)
            
            if len(recommendations) >= top_n:
                break
                
    return list(recommendations)[:top_n]

def healthy_recommendations(user_items, dataset_vocab, top_n=3):
    """
    A contextual recommender for when 'healthy' intent is detected.
    Prioritizes fruits, vegetables, and low-fat items not currently in the cart.
    """
    healthy_keywords = ['fruit', 'vegetable', 'salad', 'yogurt', 'milk', 'water', 'chicken', 'fish']
    
    healthy_options = [item for item in dataset_vocab 
                      if any(hw in item for hw in healthy_keywords) 
                      and item not in user_items]
                      
    return healthy_options[:top_n]
