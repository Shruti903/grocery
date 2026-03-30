import streamlit as st
import pandas as pd
import sys
import os
from io import StringIO
import traceback

# Import Utility Modules
from utils import preprocessing
from utils import nlp_utils
from utils import recommender

# Required to access static files appropriately
sys.path.append(os.path.dirname(__file__))

# --- App Configuration & Styling ---
st.set_page_config(
    page_title="Smart Grocery Generator",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .highlight-match { color: #f0f2f6; background-color: #28a745; border-radius: 4px; padding: 2px 4px; font-weight: bold; }
    .highlight-intent { color: #ffffff; background-color: #007bff; border-radius: 4px; padding: 2px 4px; font-weight: bold; }
    .highlight-recommend { color: #ffffff; background-color: #ffc107; border-radius: 4px; padding: 2px 4px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- Data Caching Functions ---
@st.cache_data(show_spinner=False)
def load_data_and_rules():
    df = recommender.load_dataset('dataset/Groceries_dataset.csv')
    freq = recommender.get_item_frequencies(df)
    unique_items = df['itemDescription'].unique() if not df.empty else []
    # Using low metrics for fast initialization
    rules = recommender.build_association_rules(df, min_support=0.001, min_confidence=0.05)
    return df, freq, unique_items, rules

# Load backend elements efficiently
with st.spinner("Initializing AI Engine and loading context rules..."):
    df, freq_series, vocab, association_rules = load_data_and_rules()

# --- Application Layout ---
st.title("🛒 Smart Grocery List Generator")
st.markdown("*Use Natural Language to instantly structure a context-aware shopping list.*")

# --- Initializing Session State ---
if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ""
if 'shopping_list' not in st.session_state:
    st.session_state['shopping_list'] = set()

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    smart_mode = st.toggle("Brain Mode (Advanced NLP)", value=True, help="Toggle AI Semantic understanding.")
    enable_recommendations = st.toggle("Enable Smart Recommendations", value=True)
    enable_synonyms = st.toggle("Enable Synonym Expansion", value=True)
    
    st.divider()
    
    st.subheader("📊 Dataset Diagnostics")
    st.metric(label="Total Unique Items", value=len(vocab))
    st.metric(label="Association Rules Mined", value=len(association_rules))
    
    with st.expander("Preview Dataset"):
        st.dataframe(df.head(10))
        
    st.divider()
    
    st.subheader("🔍 NLP Pipeline Explained")
    st.caption("1. **Tokenization & LEM**: Input is normalized to its root words.")
    st.caption("2. **Entity Extraction**: NLTK tags syntax blocks to extract items.")
    st.caption("3. **Semantic Matching**: TF-IDF models distance between extracted item and actual dataset item.")
    st.caption("4. **Apriori Algorithm**: Recommend items frequently purchased together.")

# --- Main Interaction ---
col1, col2 = st.columns([2, 1])

with col1:
    user_text = st.text_area(
        "What are we shopping for? Describe your meals or items:",
        placeholder="e.g., I want to cook pasta and make a healthy breakfast with juice...",
        value=st.session_state['user_input'],
        height=150
    )
    
    col_submit, col_clear = st.columns([1, 4])
    submit_btn = col_submit.button("Generate List", type="primary", use_container_width=True)
    if col_clear.button("Clear Input", use_container_width=True):
        st.session_state['user_input'] = ""
        st.session_state['shopping_list'] = set()
        st.rerun()

if submit_btn and user_text.strip():
    with st.spinner("Analyzing intent and building list..."):
        try:
            # 1. Preprocessing
            clean_text = preprocessing.clean_text(user_text)
            intents = preprocessing.detect_intent(clean_text)
            
            # Check for hardcoded semantic mappings (Pasta, Breakfast, etc)
            mapped_intent_items, detected_contexts = nlp_utils.get_context_items(clean_text)
            
            # 2. Basic vs Advanced NLP
            if smart_mode:
                if mapped_intent_items:
                    # If an intent is confidently detected, use its exact mapped items 
                    # rather than blindly extracting nouns.
                    extracted_entities = mapped_intent_items
                else:
                    # Extract nouns and process text heavily as fallback
                    extracted_entities = nlp_utils.extract_entities(clean_text)
                
                # Expand synonyms with NLTK
                if enable_synonyms:
                    expanded_entities = []
                    for ent in extracted_entities:
                        expanded_entities.extend(nlp_utils.expand_synonyms(ent, vocab))
                    # Remove duplicates and add original entities
                    extracted_entities = list(set(extracted_entities + expanded_entities))
                    
                # TF-IDF Matching
                matched_dict = nlp_utils.semantic_match(extracted_entities, vocab, threshold=0.15)
                final_matched_items = list(matched_dict.keys())
                confidence_scores = list(matched_dict.values())
                avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
                
            else:
                # Basic literal matching
                tokens = preprocessing.tokenize_and_lemmatize(clean_text)
                final_matched_items = [v for v in vocab if any(t in v.lower() for t in tokens)]
                avg_confidence = 1.0 # Literal matches are 100%
            
            # Save matched to session list to persist state during interaction
            st.session_state['shopping_list'].update(final_matched_items)
            
            # 3. Generating Recommendations
            recommendations = []
            if enable_recommendations and final_matched_items:
                recommendations = recommender.recommend_items(final_matched_items, association_rules)
            
            healthy_recs = []
            if intents.get('healthy'):
                healthy_recs = recommender.healthy_recommendations(final_matched_items, vocab)
            
            # --- Rendering Output ---
            st.subheader("🏷️ Processed Insights")
            
            k1, k2, k3 = st.columns(3)
            
            # Combine basic intents + dictionary contexts
            active_intents = [k.capitalize() for k, v in intents.items() if v]
            if detected_contexts:
                for dc in detected_contexts:
                    if dc.capitalize() not in active_intents:
                        active_intents.append(dc.capitalize())
            
            k1.metric("Items Matched", len(final_matched_items))
            k2.metric("Intents Detected", ", ".join(active_intents) if active_intents else "None")
            if smart_mode:
                k3.metric("Avg Match Confidence", f"{avg_confidence:.0%}")
            
            # Highlight Output Section
            with st.expander("Under the Hood - Keyword Extractions", expanded=False):
                st.write("**Target Entities & Mappings:**")
                if smart_mode and extracted_entities:
                    if mapped_intent_items:
                        st.info(f"**Exact Mapping Used for Context:** {', '.join(detected_contexts).upper()}")
                    mapped_view = {k: v for k,v in zip(final_matched_items, confidence_scores)} if final_matched_items else {}
                    st.json(mapped_view)
                else:
                    st.caption(f"Basic matching found {len(final_matched_items)} exact occurrences in dataset.")
            
            st.divider()
            
            # Categorized Output View
            shop_col, act_col = st.columns([1.5, 1])
            
            with shop_col:
                st.subheader("📋 Your Digital Grocery List")
                
                # We use a set form rendering to manage interactive checkboxes 
                # meaning "Shopping Mode"
                all_display_items = list(st.session_state['shopping_list'])
                sorted_items = sorted(all_display_items)
                
                if not sorted_items:
                    st.info("No concrete items detected matching our catalog. Try being specific (e.g., 'Apple', 'Milk').")
                
                check_status = {}
                for idx, item in enumerate(sorted_items):
                    check_status[item] = st.checkbox(f"{item.capitalize()}", key=f"item_{idx}")
                    
                checked_items = [k for k, v in check_status.items() if v]
                if st.button("Mark Selected as Purchased (Remove)"):
                    for i in checked_items:
                        st.session_state['shopping_list'].discard(i)
                    st.rerun()

            with act_col:
                st.subheader("💡 Smart Suggestions")
                if enable_recommendations:
                    if recommendations:
                        st.success("**Frequently Bought Together:**")
                        for rec in recommendations:
                            if st.button(f"➕ Add {rec.capitalize()}", key=f"rec_{rec}"):
                                st.session_state['shopping_list'].add(rec)
                                st.rerun()
                    else:
                        st.caption("No strong apriori associations found for these items.")
                
                if healthy_recs and intents.get('healthy'):
                    st.info("**Health-Conscious Substitutions/Additions:**")
                    for hr in healthy_recs:
                        if st.button(f"➕ Add {hr.capitalize()} 🥗", key=f"hr_{hr}"):
                            st.session_state['shopping_list'].add(hr)
                            st.rerun()
                            
            # Export function
            st.divider()
            if st.session_state['shopping_list']:
                csv_data = StringIO()
                csv_data.write("Item,Purchased\n")
                for item in list(st.session_state['shopping_list']):
                    csv_data.write(f"{item.capitalize()},False\n")
                st.download_button(
                    label="📤 Export Groceries to CSV",
                    data=csv_data.getvalue(),
                    file_name="smart_grocery_list.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")
            st.code(traceback.format_exc())
            
elif submit_btn and not user_text.strip():
    st.warning("Please provide a description of what you're shopping for!")

# Decorative Bar Chart natively for popular items 
with col2:
    st.subheader("🔥 Top Requested Groceries Overall")
    # Take top 10 items
    if 'freq_series' in locals() and not freq_series.empty:
        top_10 = freq_series.head(10).reset_index()
        top_10.columns = ['Item', 'Frequency']
        # Streamlit native chart
        st.bar_chart(top_10.set_index('Item'), color="#ff4b4b", height=350)
        st.caption("Data aggregated from historical member receipts (38,000+ purchases).")
