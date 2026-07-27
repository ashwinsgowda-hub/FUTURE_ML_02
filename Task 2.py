import os
import re
import string
import numpy as np
import pandas as pd
import streamlit as st
import kagglehub

# Machine Learning Framework
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. KAGGLE DATASET DOWNLOAD & PREPROCESSING
# ==========================================
@st.cache_data
def fetch_kaggle_dataset():
    """
    Downloads and loads the 'suraj520/customer-support-ticket-dataset' directly 
    from Kaggle using official kagglehub tooling.
    Pure data-fetching function with NO Streamlit UI elements inside.
    """
    # Download dataset files directly from Kaggle
    dataset_dir = kagglehub.dataset_download("suraj520/customer-support-ticket-dataset")
    csv_path = os.path.join(dataset_dir, "customer_support_tickets.csv")
    
    df = pd.read_csv(csv_path)

    # Locate relevant Kaggle columns
    desc_col = 'Ticket Description' if 'Ticket Description' in df.columns else 'Ticket Subject'
    type_col = 'Ticket Type' if 'Ticket Type' in df.columns else 'Category'
    prio_col = 'Ticket Priority' if 'Ticket Priority' in df.columns else 'Priority'

    df = df.dropna(subset=[desc_col, type_col]).copy()

    # Combine Subject + Description for improved context
    if 'Ticket Subject' in df.columns and 'Ticket Description' in df.columns:
        df['Full_Text'] = df['Ticket Subject'].fillna('') + " - " + df['Ticket Description'].fillna('')
    else:
        df['Full_Text'] = df[desc_col].astype(str)

    df['Ticket Type'] = df[type_col].astype(str)

    # Priority mapping logic from Kaggle dataset
    if prio_col in df.columns and df[prio_col].nunique() > 1:
        df['Ticket Priority'] = df[prio_col].astype(str)
    else:
        def assign_prio(txt):
            txt_lower = str(txt).lower()
            if any(w in txt_lower for w in ['urgent', 'fraud', 'error', 'immediately', 'crash', 'locked', 'critical', 'failed']):
                return 'High'
            elif any(w in txt_lower for w in ['refund', 'charge', 'issue', 'delay', 'dispute', 'problem']):
                return 'Medium'
            return 'Low'
        df['Ticket Priority'] = df['Full_Text'].apply(assign_prio)

    return df


# Standard NLP Stopwords List
STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'as', 'at', 
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'could', 'did', 'do', 
    'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'has', 'have', 'having', 
    'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it', 
    'its', 'itself', 'just', 'me', 'more', 'most', 'my', 'myself', 'no', 'nor', 'of', 'off', 'on', 'once', 'only', 
    'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', 'she', 'should', 'so', 'some', 
    'such', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 
    'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'we', 'were', 'what', 'when', 
    'where', 'which', 'while', 'who', 'whom', 'why', 'with', 'would', 'you', 'your', 'yours', 'yourself'
}

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = [word for word in text.split() if word not in STOPWORDS and len(word) > 2]
    return " ".join(tokens)


def filter_rare_classes(X, y, min_samples=2):
    class_counts = y.value_counts()
    valid_classes = class_counts[class_counts >= min_samples].index
    mask = y.isin(valid_classes)
    return X[mask], y[mask]


# ==========================================
# 2. MODEL TRAINING ON KAGGLE DATASET
# ==========================================
@st.cache_resource
def train_kaggle_models():
    df = fetch_kaggle_dataset()
    df['Cleaned_Text'] = df['Full_Text'].apply(clean_text)

    # Feature Extraction
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), sublinear_tf=True)
    X_tfidf = vectorizer.fit_transform(df['Cleaned_Text'])

    # Category Classification Model
    X_cat_f, y_cat_f = filter_rare_classes(pd.Series(range(X_tfidf.shape[0])), df['Ticket Type'], min_samples=2)
    cat_model = LogisticRegression(max_iter=1000, class_weight='balanced')
    cat_model.fit(X_tfidf[X_cat_f.values], y_cat_f)

    # Priority Classification Model
    X_prio_f, y_prio_f = filter_rare_classes(pd.Series(range(X_tfidf.shape[0])), df['Ticket Priority'], min_samples=2)
    prio_model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced_subsample')
    prio_model.fit(X_tfidf[X_prio_f.values], y_prio_f)

    return vectorizer, cat_model, prio_model, len(df)


# ==========================================
# 3. STREAMLIT WEB INTERFACE
# ==========================================
st.set_page_config(page_title="Support Ticket Classifier", page_icon="🎫")

st.title("🎫 Kaggle Support Ticket Classifier & Prioritizer")
st.write("Auto-downloading and training live on the **Kaggle Customer Support Ticket Dataset** (`suraj520/customer-support-ticket-dataset`).")

# Reset / Re-train Button
if st.sidebar.button("Re-train & Reset Cache"):
    st.cache_resource.clear()
    st.cache_data.clear()
    st.rerun()

try:
    with st.spinner("Downloading Dataset from Kaggle & Training Models..."):
        vectorizer, cat_model, prio_model, row_count = train_kaggle_models()
    st.sidebar.success(f"✅ Trained on **{row_count:,}** rows directly from Kaggle!")
except Exception as e:
    st.error(f"⚠️ Error during training: {e}")
    st.stop()

user_ticket = st.text_area(
    "Ticket Description:", 
    placeholder="e.g., Urgent! My account is locked and software crashes every time I open it.", 
    height=120
)

if st.button("Classify Ticket", type="primary"):
    if user_ticket.strip():
        cleaned = clean_text(user_ticket)
        vec = vectorizer.transform([cleaned])

        category = cat_model.predict(vec)[0]
        priority = prio_model.predict(vec)[0]

        # Override rule for explicit high-priority keywords
        txt_lower = user_ticket.lower()
        if any(w in txt_lower for w in ['urgent', 'crash', 'crashed', 'locked', 'fraud', 'immediately', 'down', 'unauthorized', 'error code']):
            priority = 'High'

        st.subheader("Classification Results")
        col1, col2 = st.columns(2)

        with col1:
            st.metric(label="Predicted Category", value=str(category))
        with col2:
            st.metric(label="Predicted Priority", value=str(priority))

        # Class Probability Breakdown Table
        prio_probs = prio_model.predict_proba(vec)[0]
        prio_classes = prio_model.classes_

        st.write("---")
        st.write("### 📊 Priority Confidence Breakdown")
        prob_df = pd.DataFrame({'Priority Level': prio_classes, 'Confidence': [f"{p*100:.2f}%" for p in prio_probs]})
        st.dataframe(prob_df, use_container_width=True)

    else:
        st.warning("Please enter ticket text before submitting!")