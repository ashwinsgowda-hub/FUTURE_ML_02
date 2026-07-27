# 🎫 Support Ticket Classifier & Prioritizer

An end-to-end Machine Learning web application built with **Streamlit** and **scikit-learn** that automatically categorizes customer support tickets and predicts their priority levels in real-time. 

The application automatically streams and trains on the official [Kaggle Customer Support Ticket Dataset](https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset) using `kagglehub`.

---

## ✨ Features

- 🔄 **Direct Kaggle Dataset Integration**: Streams real-time ticket data using official `kagglehub` tooling without manual CSV downloads.
- 🧹 **Custom NLP Preprocessing**: Text normalization, noise removal, and custom stop-word filtering.
- ⚡ **Multi-Model Machine Learning**:
  - **Logistic Regression** (TF-IDF) for multi-class **Ticket Type** classification.
  - **Random Forest Classifier** for **Priority Level** prediction.
- 🎯 **Rule-Based Overrides**: High-priority keyword triggers for critical operational terms (e.g., *urgent*, *crash*, *locked*).
- 📊 **Probability Breakdown**: Displays detailed priority confidence percentages for incoming tickets.
- ⚡ **Streamlit Caching**: Optimized model training and dataset loading using `@st.cache_resource` and `@st.cache_data`.

---

## 🛠️ Installation & Setup

### 1. Prerequisites
Ensure you have **Python 3.10+** installed on your system.

### 2. Clone the Repository
```bash
git clone [https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY_NAME.git)
cd YOUR_REPOSITORY_NAME
