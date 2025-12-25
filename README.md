# Fake News Prediction & Credibility Analysis

An end-to-end Natural Language Processing (NLP) and Machine Learning project to classify news articles as Fake or Real using text analytics, multiple ML models, explainability techniques, and visual insights.

---

## Problem Statement
The rapid spread of misinformation on digital platforms poses serious social and economic risks. This project aims to automatically detect fake news articles using machine learning and NLP techniques to support content credibility assessment.

---

##  Dataset
- Source: Public Fake and Real News dataset (Kaggle)
- Data Type: News article text
- Classes:
  - 0 → Fake News
  - 1 → Real News

---

## Solution Overview
- Text preprocessing and cleaning (punctuation, URLs, stopwords)
- Exploratory Data Analysis (EDA) to identify linguistic patterns
- TF-IDF feature extraction with n-grams
- Multiple ML models:
  - Logistic Regression
  - Naive Bayes
  - Linear SVM
  - Random Forest
- Model comparison and selection
- Hyperparameter tuning and cross-validation
- Explainability and error analysis
- Visualization for performance and insights

---

## Key EDA Insights
- Fake news articles tend to be shorter and more compact
- Real news articles show richer vocabulary and longer structure
- Class imbalance influenced the use of class-weighted models
- N-gram patterns helped improve classification accuracy

---

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- NLP (TF-IDF, n-grams)
- Matplotlib, Seaborn

---

## Results
- Achieved ~95% accuracy using Logistic Regression / Linear SVM
- Linear models outperformed ensemble models for sparse text features
- Explainability analysis identified influential words contributing to predictions

---

## How to Run the Project
```bash
pip install -r requirements.txt
