import os
import sys
import time
import streamlit as st
import pandas as pd
import io
import numpy as np 

# --- CONFIGURATION: Intelligent Threading & Resource Limits ---
# 1. Mac (Darwin): Strict single-threading needed to prevent "mutex" crashes.
# 2. Streamlit Cloud (Linux): We limit threads to 2 to prevent "Out of Memory" (OOM) kills 
#    and CPU thrashing on Free Tier (which usually has ~3GB RAM limits).
if sys.platform == "darwin":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
else:
    # Linux / Cloud settings (Memory Safety)
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    os.environ["NUMEXPR_NUM_THREADS"] = "2"
    # Allow tokenizers parallelism but keep it modest
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="BERTopic Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Check Environment (Debug Help)
# -----------------------------------------------------------------------------
if sys.platform == "darwin":
    if os.environ.get('OBJC_DISABLE_INITIALIZE_FORK_SAFETY') != 'YES':
        st.error("""
        🛑 **Critical Error: App launched incorrectly (Local Mac)**
        
        You are running on macOS, which requires a specific security flag to be disabled.
        Please stop this app and run it using the `run_app.sh` script provided.
        """)
        st.stop()

# -----------------------------------------------------------------------------
# Language & Translations
# -----------------------------------------------------------------------------
if 'lang' not in st.session_state:
    st.session_state['lang'] = 'en'

def toggle_language():
    st.session_state['lang'] = 'zh' if st.session_state['lang'] == 'en' else 'en'

TRANS = {
    'title': { 'en': "🧠 BERTopic Interactive Explorer", 'zh': "🧠 BERTopic 交互式探索器" },
    'desc': {'en': "Advanced Topic Modeling with BERTopic.", 'zh': "BERTopic 高级主题建模。"},
    'sidebar_config': {'en': "Configuration", 'zh': "配置"},
    'remove_stopwords': {'en': "Remove Stopwords (English)", 'zh': "移除停用词 (英文)"},
    'lemmatize': {'en': "Combine Variations (Lemmatize)", 'zh': "合并词形变体 (Lemmatize)"},
    'lemmatize_help': {'en': "Converts words to base form (e.g., 'students' -> 'student'). Slower but cleaner.", 'zh': "将单词转换为基本形式（例如 'students' -> 'student'）。速度较慢但结果更整洁。"},
    'data_loading': {'en': "Data Loading", 'zh': "数据加载"},
    'upload_csv': {'en': "Upload CSV", 'zh': "上传 CSV"},
    'train_btn': {'en': "🚀 Train BERTopic Model", 'zh': "🚀 训练 BERTopic 模型"},
    'status_start': {'en': "Starting Process...", 'zh': "正在启动流程..."},
    'step_1': {'en': "⚙️ [1/3] Loading AI Model...", 'zh': "⚙️ [1/3] 加载 AI 模型..."},
    'step_2': {'en': "🏃 [2/3] Embedding & Clustering (This takes the longest)...", 'zh': "🏃 [2/3] 嵌入与聚类 (这一步最耗时)..."},
    'train_complete': {'en': "Complete! Time: {:.2f}s", 'zh': "完成! 耗时: {:.2f} 秒"},
    'results_header': {'en': "Results Analysis", 'zh': "结果分析"},
    'upload_prompt': {'en': "Please upload a CSV file to begin.", 'zh': "请上传 CSV 文件以开始。"},
    'no_topics_warning': {
        'en': "⚠️ No topics were found! Everything was classified as outliers (-1). Try decreasing 'Min Topic Size' or adding more data.",
        'zh': "⚠️ 未发现任何主题！所有内容都被归类为离群值 (-1)。请尝试减小“最小主题大小”或添加更多数据。"
    },
    'viz_error': {'en': "Visualization not available: {}", 'zh': "无法生成可视化: {}"},
    'help_info_title': {'en': "ℹ️ How to interpret this table", 'zh': "ℹ️ 如何解读此表"},
    'help_info_text': {
        'en': "**Topic:** The ID of the topic. -1 refers to 'outliers' (noise).\n**Count:** Documents in this topic.\n**Name:** Keywords representing the topic.",
        'zh': "**Topic:** 主题 ID。-1 代表“离群值”（噪音）。\n**Count:** 文档数量。\n**Name:** 代表该主题的关键词。"
    },
    'help_dist_title': {'en': "ℹ️ How to interpret the Distance Map", 'zh': "ℹ️ 如何解读距离图"},
    'help_dist_text': {
        'en': "**Circles:** Topics.\n**Distance:** Closer circles = Similar meanings.",
        'zh': "**圆圈:** 主题。\n**距离:** 圆圈越近 = 含义越相似。"
    },
    'help_bar_title': {'en': "ℹ️ How to interpret the Bar Chart", 'zh': "ℹ️ 如何解读条形图"},
    'help_bar_text': {
        'en': "Shows distinct keywords for each topic based on c-TF-IDF score.",
        'zh': "基于 c-TF-IDF 分数显示每个主题的独特关键词。"
    },
    'help_heat_title': {'en': "ℹ️ How to interpret the Similarity Heatmap", 'zh': "ℹ️ 如何解读相似度热力图"},
    'help_heat_text': {
        'en': "Shows similarity between topics. Dark blue = High similarity.",
        'zh': "显示主题间的相似度。深蓝色 = 高相似度。"
    }
}

def t(key):
    return TRANS.get(key, {}).get(st.session_state['lang'], "Missing")

# -----------------------------------------------------------------------------
# Styling Helpers
# -----------------------------------------------------------------------------
def style_fig(fig):
    if fig:
        fig.update_layout(hoverlabel=dict(bgcolor="#333333", font_color="#4b8bf5", font_family="sans-serif", bordercolor="#4b8bf5"))
    return fig

# -----------------------------------------------------------------------------
# Caching High-Cost Resources (Speed Boost)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_embedding_model():
    """Loads the sentence transformer model once and keeps it in memory."""
    from sentence_transformers import SentenceTransformer
    # 'all-MiniLM-L6-v2' is the fastest accurate model for CPU
    return SentenceTransformer('all-MiniLM-L6-v2')

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.button("🌐 English / 中文", on_click=toggle_language)
st.sidebar.title(t('sidebar_config'))

remove_stopwords = st.sidebar.checkbox(t('remove_stopwords'), value=True)
use_lemmatization = st.sidebar.checkbox(t('lemmatize'), value=False)

# FAST MODE TOGGLE
fast_mode = st.sidebar.checkbox(t('fast_mode'), value=True, help=t('fast_mode_help'))

docs = []
uploaded_file = st.sidebar.file_uploader(t('upload_csv'), type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        text_col = st.sidebar.selectbox("Text Column", df.columns)
        
        # Data Cleaning
        df = df.dropna(subset=[text_col])
        df = df[df[text_col].astype(str).str.strip() != '']
        df = df.reset_index(drop=True)
        
        if len(df) == 0:
            st.error("Error: No valid text data found.")
        else:
            # SAMPLING LOGIC
            if fast_mode and len(df) > 5000:
                df = df.sample(5000, random_state=42)
                st.sidebar.warning(f"⚡ Fast Mode: Using 5,000 random samples out of {len(uploaded_file.getvalue())} bytes.")
            
            docs = df[text_col].astype(str).tolist()
            st.sidebar.success(f"Loaded {len(docs)} docs")
    except Exception as e:
        st.sidebar.error(f"Error reading CSV: {e}")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def get_lemmatizer_analyzer():
    from sklearn.feature_extraction.text import CountVectorizer
    try:
        import nltk
        from nltk.stem import WordNetLemmatizer
        try:
            nltk.data.find('corpora/wordnet')
            nltk.data.find('corpora/omw-1.4')
        except LookupError:
            with st.spinner("Downloading NLTK data (WordNet)... this happens once."):
                nltk.download('wordnet')
                nltk.download('omw-1.4')
    except ImportError:
        return None  
    lemmatizer = WordNetLemmatizer()
    analyzer = CountVectorizer(stop_words="english").build_analyzer()
    def lemmatized_words(doc):
        return [lemmatizer.lemmatize(w) for w in analyzer(doc)]
    return lemmatized_words

# -----------------------------------------------------------------------------
# Main App Logic
# -----------------------------------------------------------------------------
st.title(t('title'))

# Model Params
language = st.sidebar.selectbox("Language", ["english", "multilingual"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("**Step 1: Discovery**")
min_topic_size = st.sidebar.number_input("Min Topic Size", min_value=2, value=5, step=1)

st.sidebar.markdown("**Step 2: Reduction**")
auto_topics = st.sidebar.checkbox("Auto Reduce Topics", value=True)
if auto_topics:
    nr_topics = "auto"
else:
    nr_topics = st.sidebar.slider("Target Max Topics", 5, 300, 20)

st.sidebar.markdown("---")
auto_adjust_params = st.sidebar.checkbox("Auto-adjust parameters for small data", value=True)

if st.button(t('train_btn'), type="primary", disabled=(not docs)):
    start_time = time.time()
    with st.status(t('status_start'), expanded=True) as status:
        try:
            # 1. Load Embedding Model (Cached)
            st.write(t('step_1'))
            import torch
            # Only set threads if on Mac
            if sys.platform == "darwin":
                torch.set_num_threads(1)
            
            from bertopic import BERTopic
            from umap import UMAP
            from hdbscan import HDBSCAN
            from sklearn.feature_extraction.text import CountVectorizer 

            # Load Cached Model
            embedding_model = load_embedding_model()

            # 2. Configure Vectorizer
            if use_lemmatization:
                custom_analyzer = get_lemmatizer_analyzer()
                if custom_analyzer:
                    vectorizer_model = CountVectorizer(analyzer=custom_analyzer)
                else:
                    st.warning("NLTK missing, skipping lemmatization.")
                    vectorizer_model = CountVectorizer(stop_words="english") if remove_stopwords else None
            else:
                vectorizer_model = CountVectorizer(stop_words="english") if remove_stopwords else None

            # 3. Configure Sub-models
            n_samples = len(docs)
            
            # Safety checks
            safe_min_topic_size = min_topic_size
            if min_topic_size >= n_samples:
                safe_min_topic_size = max(2, n_samples - 1)

            # UMAP Settings
            n_neighbors_val = 15
            n_components_val = 5
            
            if auto_adjust_params and n_samples < 20:
                n_neighbors_val = max(2, min(15, n_samples - 1))
                n_components_val = max(2, min(5, n_samples - 2))
            
            umap_model = UMAP(
                n_neighbors=n_neighbors_val, 
                n_components=n_components_val, 
                min_dist=0.0, 
                metric='cosine', 
                low_memory=True, # Optimized for Streamlit Cloud Memory Limits
                n_jobs=1 
            )
            
            hdbscan_model = HDBSCAN(
                min_cluster_size=safe_min_topic_size, 
                metric='euclidean', 
                cluster_selection_method='eom', 
                prediction_data=True, 
                core_dist_n_jobs=1
            )

            topic_model = BERTopic(
                language=language,
                nr_topics=nr_topics if nr_topics == "auto" else int(nr_topics),
                min_topic_size=safe_min_topic_size,
                vectorizer_model=vectorizer_model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                embedding_model=embedding_model, # Use cached model
                verbose=True
            )

            # 4. Fit
            st.write(t('step_2'))
            
            clean_docs = [str(d) for d in docs]
            
            try:
                topics, probs = topic_model.fit_transform(clean_docs)
            except ValueError as ve:
                st.error(f"Analysis failed: {ve}")
                st.stop()
            
            # Flatten & Check
            topics_list = np.array(topics).flatten().tolist()
            topics_list = [int(t) for t in topics_list]
            
            if len(clean_docs) != len(topics_list):
                min_len = min(len(clean_docs), len(topics_list))
                clean_docs = clean_docs[:min_len]
                topics_list = topics_list[:min_len]

            # 5. Update Keywords
            try:
                topic_model.update_topics(clean_docs, topics_list, vectorizer_model=vectorizer_model)
            except Exception as e:
                st.warning(f"Keyword fine-tuning skipped: {e}")

            # Store
            st.session_state['model'] = topic_model
            st.session_state['docs'] = clean_docs
            st.session_state['topics'] = topics_list
            
            st.success(t('train_complete').format(time.time() - start_time))
            status.update(label="Done", state="complete", expanded=False)

        except Exception as e:
            st.error(f"Error: {str(e)}")

# -----------------------------------------------------------------------------
# Visualization Section
# -----------------------------------------------------------------------------
if 'model' in st.session_state:
    model = st.session_state['model']
    topic_info = model.get_topic_info()
    real_topic_count = len(topic_info) - 1 
    has_topics = real_topic_count > 0
    
    st.divider()
    st.header(t('results_header'))
    
    import plotly.express as px

    tab1, tab2, tab3, tab4 = st.tabs(["Topic Info", "Distance Map", "Bar Chart", "Heatmap"])
    
    with tab1:
        with st.expander(t('help_info_title')): st.markdown(t('help_info_text'))
        st.dataframe(topic_info, use_container_width=True)
        t_ids = topic_info['Topic'].values
        sel_t = st.selectbox("Explore Topic", t_ids)
        if sel_t is not None: st.write(model.get_topic(sel_t))

    with tab2:
        with st.expander(t('help_dist_title')): st.markdown(t('help_dist_text'))
        if not has_topics: st.warning(t('no_topics_warning'))
        elif real_topic_count < 4: st.info("Not enough topics for Distance Map.")
        else:
            try:
                fig = model.visualize_topics()
                st.plotly_chart(style_fig(fig), use_container_width=True)
            except Exception as e: st.warning(t('viz_error').format(e))

    with tab3:
        with st.expander(t('help_bar_title')): st.markdown(t('help_bar_text'))
        if not has_topics: st.warning(t('no_topics_warning'))
        else:
            try:
                fig = model.visualize_barchart(top_n_topics=10)
                st.plotly_chart(style_fig(fig), use_container_width=True)
            except Exception as e: st.warning(t('viz_error').format(e))

    with tab4:
        with st.expander(t('help_heat_title')): st.markdown(t('help_heat_text'))
        if not has_topics: st.warning(t('no_topics_warning'))
        else:
            try:
                fig = model.visualize_heatmap()
                st.plotly_chart(style_fig(fig), use_container_width=True)
            except Exception as e: st.warning(t('viz_error').format(e))
elif not docs:
    st.info(t('upload_prompt'))
