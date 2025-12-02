import streamlit as st
import pandas as pd
import numpy as np
import re
import concurrent.futures

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

# Setup NLTK (Download quietly)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')

# --- CONFIGURATION ---
st.set_page_config(page_title="AI Spam Detection", layout="wide")

# --- UTILS ---
@st.cache_data
def process_dataframe(df, target_col):
    """
    Heavy lifting: Filters, Translation, etc.
    Cached by Streamlit for performance.
    """
    processed_df = df.copy()
    processed_df['Status'] = 'Non-Spam'
    processed_df['Reason'] = 'Valid'
    # Initialize Translation as empty string
    processed_df['Translation'] = ""
    
    # 1. Filters
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
    
    # Crypto Check (Case Insensitive)
    crypto_pattern = r'Cryptaxbot|bitcoin'
    crypto_mask = (processed_df['Status'] == 'Non-Spam') & text_series.str.contains(crypto_pattern, case=False, regex=True, na=False)
    processed_df.loc[crypto_mask, 'Status'] = 'Spam'
    processed_df.loc[crypto_mask, 'Reason'] = 'Crypto Keyword'

    # Suspicious Keywords Check (Marijuana, Casino, Bots)
    suspicious_pattern = r'marijuana|xevil|casino|captchas|recaptcha'
    suspicious_mask = (processed_df['Status'] == 'Non-Spam') & text_series.str.contains(suspicious_pattern, case=False, regex=True, na=False)
    processed_df.loc[suspicious_mask, 'Status'] = 'Spam'
    processed_df.loc[suspicious_mask, 'Reason'] = 'Suspicious Keyword'

    # Email Check
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    email_mask = (processed_df['Status'] == 'Non-Spam') & text_series.str.contains(email_pattern, regex=True, na=False)
    processed_df.loc[email_mask, 'Status'] = 'Spam'
    processed_df.loc[email_mask, 'Reason'] = 'Contains Email'
    
    # -----------------------------------
    
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

    # 2. Translation (Parallel)
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
                return idx, translate(txt[:500], 'en')
            except:
                return idx, None

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(do_trans, to_translate))
            
        for idx, trans in results:
            if trans:
                processed_df.at[idx, 'Translation'] = trans
                
    return processed_df

def get_topics(df_subset, target_col, n_topics=3):
    if df_subset.empty: return [], None
    try:
        stop_words = set(stopwords.words('english'))
        clean_docs = []
        for index, row in df_subset.iterrows():
            text_content = row['Translation'] if row['Translation'] else row[target_col]
            t = re.sub(r'[^a-zA-Z\s]', '', str(text_content).lower())
            tokens = [w for w in word_tokenize(t) if w not in stop_words and len(w) > 3]
            clean_docs.append(" ".join(tokens))
            
        if not "".join(clean_docs).strip(): return []

        vec = CountVectorizer(max_features=500, stop_words='english')
        X = vec.fit_transform(clean_docs)
        # Using a fixed seed for reproducibility
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42) 
        lda.fit(X)
        
        feature_names = vec.get_feature_names_out()
        topics = []
        for topic in lda.components_:
            top = [feature_names[i] for i in topic.argsort()[:-8:-1]]
            topics.append(", ".join(top))
            
        # Generate WordCloud
        wc = WordCloud(width=800, height=300, background_color='white', max_words=100).generate(" ".join(clean_docs))
        return topics, wc
    except:
        return [], None

# --- MAIN APP ---
st.title("🛡️ AI Spam Detection Dashboard")

uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    cols = df.columns.tolist()
    
    target_col = st.selectbox("Select Text Column for Analysis", cols)
    
    if st.button("Run Analysis"):
        with st.spinner("Processing... Filtering, Translating, and Analyzing."):
            # Clear previous cache to ensure fresh analysis if data changes
            st.cache_data.clear() 
            processed_df = process_dataframe(df, target_col)
            st.session_state['data'] = processed_df
            st.session_state['target_col'] = target_col
            st.success("Analysis Complete!")

if 'data' in st.session_state:
    data = st.session_state['data']
    target_col = st.session_state['target_col']
    
    # Tabs
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
        st.caption("Check the box to manually mark items as spam. Changes apply immediately.")
        st.caption("💡 **Tip:** Double-click on any cell to expand and view the full text.")
        
        non_spam_df = data[data['Status'] == 'Non-Spam']
        
        review_df = non_spam_df.copy()
        review_df['Mark as Spam'] = False
        
        # Reorder columns for display
        display_cols = ['Mark as Spam', 'Translation', target_col] + [c for c in review_df.columns if c not in ['Mark as Spam', 'Translation', target_col, 'Status', 'Reason']]
        
        # Streamlit's data_editor provides the table and editing function
        edited_df = st.data_editor(
            review_df[display_cols],
            column_config={
                "Mark as Spam": st.column_config.CheckboxColumn(
                    "Mark Spam",
                    help="Check to move this item to Spam category",
                    default=False,
                ),
                "Translation": st.column_config.TextColumn(
                    "Translation",
                    width="medium",
                    help="English translation of non-ASCII content"
                ),
                target_col: st.column_config.TextColumn(
                    f"{target_col} (Original)",
                    width="large", 
                    help="Original content from CSV"
                )
            },
            num_rows="fixed",
            height=600,
            use_container_width=True 
        )
        
        if st.button("Update Marked Items", key="update_marked"):
            to_mark = edited_df[edited_df['Mark as Spam'] == True].index
            
            # Update session state
            st.session_state['data'].loc[to_mark, 'Status'] = 'Spam'
            st.session_state['data'].loc[to_mark, 'Reason'] = 'Manual User Mark'
            st.rerun()

    # --- TAB 2: Manually Flagged (NEW) ---
    with tab_manual:
        st.markdown("#### Manage Manually Flagged Items")
        st.caption("These items were manually marked as spam by you. Check the box to restore them to Non-Spam.")
        
        # Filter for items marked manually
        manual_df = data[data['Reason'] == 'Manual User Mark']
        
        if not manual_df.empty:
            restore_df = manual_df.copy()
            restore_df['Restore'] = False
            
            # Columns to display
            display_cols_man = ['Restore', 'Translation', target_col]
            display_cols_man += [c for c in restore_df.columns if c not in display_cols_man and c not in ['Status', 'Reason', 'Mark as Spam']]
            
            edited_manual_df = st.data_editor(
                restore_df[display_cols_man],
                column_config={
                    "Restore": st.column_config.CheckboxColumn(
                        "Restore",
                        help="Check to move this item back to Non-Spam",
                        default=False,
                    ),
                    "Translation": st.column_config.TextColumn(
                        "Translation",
                        width="medium",
                    ),
                    target_col: st.column_config.TextColumn(
                        f"{target_col} (Original)",
                        width="large", 
                    )
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
                else:
                    st.info("No items selected to restore.")
        else:
            st.info("No manually marked spam items found.")

    # --- TAB 3: Stats ---
    with tab_stats:
        col1, col2 = st.columns([2, 1])
        
        spam_df = data[data['Status'] == 'Spam']
        counts = spam_df['Reason'].value_counts()
        
        with col1:
            st.subheader("Spam Categories Distribution")
            
            # Using Matplotlib to ensure small labels like in the desktop app
            fig, ax = plt.subplots()
            counts.plot(kind='bar', color='#FF4B4B', ax=ax)
            plt.xticks(rotation=45, ha='right', fontsize=8) # Small labels, Slanted
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig)
            
        with col2:
            st.subheader("Summary")
            total = len(data)
            spam = len(spam_df)
            st.metric("Total Rows", total)
            st.metric("Spam Detected", f"{spam}", delta=f"{spam/total:.1%}", delta_color="inverse")
            st.metric("Clean Records", total - spam)
            
            st.write("---")
            st.write("**Breakdown:**")
            st.dataframe(counts, use_container_width=True)

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