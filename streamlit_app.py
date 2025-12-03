import streamlit as st
import pandas as pd
import numpy as np
import re
import concurrent.futures
import gc  # Garbage collection for memory management

# Data Science & NLP Imports
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Translation
from mtranslate import translate

# --- CONFIGURATION ---
st.set_page_config(page_title="AI Spam Detection", layout="wide")

# --- CACHED RESOURCES ---
@st.cache_resource
def setup_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('punkt_tab')

setup_nltk()

# --- UTILS ---
@st.cache_data(max_entries=1)
def process_dataframe(df, target_col):
    """
    Optimized processing pipeline with memory safeguards.
    """
    processed_df = df.copy()
    
    # Initialize columns
    processed_df['Status'] = 'Non-Spam'
    processed_df['Reason'] = 'Valid'
    processed_df['Translation'] = ""
    
    # 1. Filters (Vectorized)
    text_series = processed_df[target_col].astype(str)
    
    # Duplicate
    dup_mask = processed_df.duplicated(subset=[target_col], keep='first')
    processed_df.loc[dup_mask, 'Status'] = 'Spam'
    processed_df.loc[dup_mask, 'Reason'] = 'Duplicate'
    
    # URL
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    url_mask = (processed_df['Status'] == 'Non-Spam') & text_series.str.contains(url_pattern, regex=True, na=False)
    processed_df.loc[url_mask, 'Status'] = 'Spam'
    processed_df.loc[url_mask, 'Reason'] = 'Contains URL'
    
    # Crypto
    crypto_pattern = r'Cryptaxbot|bitcoin'
    crypto_mask = (processed_df['Status'] == 'Non-Spam') & text_series.str.contains(crypto_pattern, case=False, regex=True, na=False)
    processed_df.loc[crypto_mask, 'Status'] = 'Spam'
    processed_df.loc[crypto_mask, 'Reason'] = 'Crypto Keyword'

    # Suspicious
    suspicious_pattern = r'marijuana|xevil|casino|captchas|recaptcha'
    suspicious_mask = (processed_df['Status'] == 'Non-Spam') & text_series.str.contains(suspicious_pattern, case=False, regex=True, na=False)
    processed_df.loc[suspicious_mask, 'Status'] = 'Spam'
    processed_df.loc[suspicious_mask, 'Reason'] = 'Suspicious Keyword'

    # Email
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    email_mask = (processed_df['Status'] == 'Non-Spam') & text_series.str.contains(email_pattern, regex=True, na=False)
    processed_df.loc[email_mask, 'Status'] = 'Spam'
    processed_df.loc[email_mask, 'Reason'] = 'Contains Email'
    
    # Short
    short_mask = (processed_df['Status'] == 'Non-Spam') & (text_series.str.split().str.len() < 2)
    processed_df.loc[short_mask, 'Status'] = 'Spam'
    processed_df.loc[short_mask, 'Reason'] = 'Too Short'
    
    # Symbols
    str_len = text_series.str.len()
    clean_len = text_series.str.count(r'[a-zA-Z0-9\s]') 
    ratio = (clean_len / str_len).fillna(1.0)
    sym_mask = (processed_df['Status'] == 'Non-Spam') & (str_len > 0) & (ratio < 0.6)
    processed_df.loc[sym_mask, 'Status'] = 'Spam'
    processed_df.loc[sym_mask, 'Reason'] = 'Excessive Symbols'

    # Optimize types
    processed_df['Status'] = processed_df['Status'].astype('category')
    processed_df['Reason'] = processed_df['Reason'].astype('category')

    # 2. Translation (Throttled)
    candidates_idx = processed_df[processed_df['Status'] == 'Non-Spam'].index
    to_translate = []
    
    for idx in candidates_idx:
        txt = str(processed_df.at[idx, target_col])
        if not all(ord(c) < 128 for c in txt if c.strip()):
            to_translate.append((idx, txt))
    
    if to_translate:
        def do_trans(item):
            idx, txt = item
            try:
                return idx, translate(txt[:300], 'en')
            except:
                return idx, None

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(do_trans, to_translate))
            
        for idx, trans in results:
            if trans:
                processed_df.at[idx, 'Translation'] = trans
    
    gc.collect()
    return processed_df

def get_topics(df_subset, target_col, n_topics=3):
    if df_subset.empty: return [], None
    
    # Sampling for speed/memory
    if len(df_subset) > 2000:
        df_subset = df_subset.sample(2000, random_state=42)
    
    try:
        stop_words = set(stopwords.words('english'))
        clean_docs = []
        for index, row in df_subset.iterrows():
            text_content = row['Translation'] if row['Translation'] else row[target_col]
            t = re.sub(r'[^a-zA-Z\s]', '', str(text_content).lower())
            tokens = [w for w in word_tokenize(t) if w not in stop_words and len(w) > 3]
            clean_docs.append(" ".join(tokens))
            
        full_text = " ".join(clean_docs)
        if not full_text.strip(): return [], None

        vec = CountVectorizer(max_features=500, stop_words='english')
        X = vec.fit_transform(clean_docs)
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(X)
        
        feature_names = vec.get_feature_names_out()
        topics = []
        for topic in lda.components_:
            top = [feature_names[i] for i in topic.argsort()[:-8:-1]]
            topics.append(", ".join(top))
            
        wc = WordCloud(width=800, height=300, background_color='white', max_words=100).generate(full_text)
        
        del vec, X, lda, clean_docs, full_text
        gc.collect()
        
        return topics, wc
    except Exception as e:
        return [], None

# --- MAIN APP UI ---
st.title("🛡️ AI Spam Detection Dashboard")

with st.sidebar:
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    if uploaded_file:
        st.info(f"File Size: {uploaded_file.size / 1024 / 1024:.2f} MB")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        cols = df.columns.tolist()
        
        col1, col2 = st.columns([3, 1])
        with col1:
            target_col = st.selectbox("Select Text Column for Analysis", cols)
        with col2:
            st.write("") 
            st.write("") 
            run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
        
        if run_btn:
            with st.spinner("Processing..."):
                st.cache_data.clear()
                processed_df = process_dataframe(df, target_col)
                st.session_state['data'] = processed_df
                st.session_state['target_col'] = target_col
                st.success("Analysis Complete!")
                gc.collect()
                
    except Exception as e:
        st.error(f"Error loading file: {e}")

if 'data' in st.session_state:
    data = st.session_state['data']
    target_col = st.session_state['target_col']
    
    tab_review, tab_manual, tab_stats, tab_ns_topics, tab_s_topics = st.tabs([
        "📝 Review Non-Spam", 
        "🚩 Manually Flagged",
        "📊 Spam Statistics", 
        "✅ Non-Spam Topics", 
        "🚫 Spam Topics"
    ])
    
    # --- TAB 1: Review ---
    with tab_review:
        st.markdown("#### Review Non-Spam Messages")
        st.caption("Tip: Double-click on any cell to view full text.")
        
        non_spam_df = data[data['Status'] == 'Non-Spam']
        review_df = non_spam_df.copy()
        review_df['Mark as Spam'] = False
        
        display_cols = ['Mark as Spam', 'Translation', target_col] + [c for c in review_df.columns if c not in ['Mark as Spam', 'Translation', target_col, 'Status', 'Reason']]
        
        edited_df = st.data_editor(
            review_df[display_cols],
            column_config={
                "Mark as Spam": st.column_config.CheckboxColumn("Mark Spam", default=False),
                "Translation": st.column_config.TextColumn("Translation", width="medium"),
                target_col: st.column_config.TextColumn(f"{target_col} (Original)", width="large")
            },
            num_rows="fixed",
            height=600,
            use_container_width=True 
        )
        
        if st.button("Confirm & Move Marked Items to Spam"):
            to_mark = edited_df[edited_df['Mark as Spam'] == True].index
            if not to_mark.empty:
                st.session_state['data'].loc[to_mark, 'Status'] = 'Spam'
                st.session_state['data'].loc[to_mark, 'Reason'] = 'Manual User Mark'
                st.success(f"Moved {len(to_mark)} items to Spam.")
                st.rerun()
            gc.collect()

    # --- TAB 2: Manually Flagged ---
    with tab_manual:
        st.markdown("#### Manage Manually Flagged Items")
        manual_df = data[data['Reason'] == 'Manual User Mark']
        
        if not manual_df.empty:
            restore_df = manual_df.copy()
            restore_df['Restore'] = False
            
            display_cols_man = ['Restore', 'Translation', target_col]
            display_cols_man += [c for c in restore_df.columns if c not in display_cols_man and c not in ['Status', 'Reason', 'Mark as Spam']]
            
            edited_manual_df = st.data_editor(
                restore_df[display_cols_man],
                column_config={
                    "Restore": st.column_config.CheckboxColumn("Restore", default=False),
                    "Translation": st.column_config.TextColumn("Translation", width="medium"),
                    target_col: st.column_config.TextColumn(f"{target_col} (Original)", width="large")
                },
                num_rows="fixed",
                height=600,
                use_container_width=True,
                key="manual_editor"
            )
            
            if st.button("Restore Selected Items"):
                to_restore = edited_manual_df[edited_manual_df['Restore'] == True].index
                if not to_restore.empty:
                    st.session_state['data'].loc[to_restore, 'Status'] = 'Non-Spam'
                    st.session_state['data'].loc[to_restore, 'Reason'] = 'Valid'
                    st.success(f"Restored {len(to_restore)} items.")
                    st.rerun()
            gc.collect()
        else:
            st.info("No manually marked spam items found.")

    # --- TAB 3: Stats ---
    with tab_stats:
        col1, col2 = st.columns([2, 1])
        spam_df = data[data['Status'] == 'Spam']
        counts = spam_df['Reason'].value_counts()
        
        with col1:
            st.subheader("Spam Categories Distribution")
            
            # --- FIXED: Check if data exists before plotting to avoid IndexError ---
            if not counts.empty and counts.sum() > 0:
                fig, ax = plt.subplots(figsize=(8, 5))
                counts.sort_values().plot(kind='barh', color='#FF4B4B', ax=ax)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.xlabel("Count")
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("No spam detected yet. Charts will appear here once spam is found.")
            
        with col2:
            st.subheader("Summary")
            total = len(data)
            spam = len(spam_df)
            st.metric("Total Records", total)
            st.metric("Spam Detected", f"{spam}", delta=f"{spam/total:.1%}", delta_color="inverse")
            st.metric("Clean Records", total - spam)
            st.write("---")
            st.write("**Breakdown:**")
            st.dataframe(counts, use_container_width=True)
        gc.collect()

    # --- TAB 4: Non-Spam Topics ---
    with tab_ns_topics:
        st.subheader("✅ Non-Spam Topic Modeling")
        ns_topics_n = st.slider("Number of Topics", 2, 10, 3, key="ns_slider")
        
        ns_df = data[data['Status'] == 'Non-Spam']
        if not ns_df.empty:
            topics, wc = get_topics(ns_df, target_col, ns_topics_n)
            if wc:
                st.image(wc.to_array(), use_container_width=True, caption="Word Cloud (Non-Spam)")
            if topics:
                for i, t in enumerate(topics):
                    st.info(f"**Topic {i+1}:** {t}")
        else:
            st.warning("No data.")
        gc.collect()

    # --- TAB 5: Spam Topics ---
    with tab_s_topics:
        st.subheader("🚫 Spam Topic Modeling")
        s_topics_n = st.slider("Number of Topics", 2, 10, 3, key="s_slider")
        
        s_df = data[data['Status'] == 'Spam']
        if not s_df.empty:
            topics, wc = get_topics(s_df, target_col, s_topics_n)
            if wc:
                st.image(wc.to_array(), use_container_width=True, caption="Word Cloud (Spam)")
            if topics:
                for i, t in enumerate(topics):
                    st.error(f"**Topic {i+1}:** {t}")
        else:
            st.warning("No spam data.")
        gc.collect()
