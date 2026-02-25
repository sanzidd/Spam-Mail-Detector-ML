# 📧 Spam Mail Detector using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=flat&logo=scikit-learn)
![NLP](https://img.shields.io/badge/NLP-Text%20Classification-blueviolet?style=flat)
![Algorithm](https://img.shields.io/badge/Algorithm-Naive%20Bayes-lightblue?style=flat)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat&logo=jupyter)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat)

A Natural Language Processing (NLP) based machine learning project that detects whether an email is **spam or not spam (ham)** using a **Multinomial Naive Bayes** classifier with **Bag of Words (CountVectorizer)** text representation. The model can predict on real custom email inputs in real time.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Technologies Used](#-technologies-used)
- [Real-Time Prediction](#-real-time-prediction)
- [Results](#-results)
- [How to Run](#-how-to-run)
- [Project Structure](#-project-structure)
- [Key Concepts](#-key-concepts)

---

## 🔍 Overview

Spam emails are a major security and productivity concern. This project builds a text classification model that automatically identifies spam emails using machine learning and NLP techniques.

The pipeline converts raw email text into numerical features using **CountVectorizer (Bag of Words)**, then trains a **Multinomial Naive Bayes** model — one of the most effective and widely used algorithms for text classification tasks.

**Target Variable:** `spam` — 1 (Spam) or 0 (Ham/Not Spam)

---

## 📂 Dataset

**File:** `emails.csv`

The dataset contains real email data with two key columns:

| Column | Description |
|--------|-------------|
| `text` | The full email subject and body text |
| `spam` | Label — 1 for spam, 0 for legitimate email |

**Data Cleaning Steps:**
- Kept only the two relevant columns (`text`, `spam`), dropping 108 unnamed NaN columns
- Removed duplicate rows
- Dropped rows with missing values in the `spam` column

---

## 🔄 Project Workflow

```
1. Data Loading
       ↓
2. Data Cleaning
   - Select only 'text' and 'spam' columns
   - Drop duplicates
   - Drop null values
       ↓
3. Feature & Target Separation
   - X = email text
   - y = spam label
       ↓
4. Train-Test Split (80% Train / 20% Test)
       ↓
5. Text Vectorization
   - CountVectorizer (Bag of Words)
   - fit_transform on train set
   - transform only on test set
       ↓
6. Model Training
   - Multinomial Naive Bayes
       ↓
7. Model Evaluation
   - Accuracy Score on test set
       ↓
8. Real-Time Prediction
   - Predict on custom email inputs
```

---

## 🛠 Technologies Used

| Tool | Purpose |
|------|---------|
| Python | Core programming language |
| Pandas | Data loading and cleaning |
| NumPy | Numerical operations |
| Seaborn | Data visualization |
| Scikit-learn | Vectorization, model training & evaluation |
| Jupyter Notebook | Development environment |

---

## 🔮 Real-Time Prediction

One of the highlights of this project is the ability to test the model on **custom email inputs** directly:

```python
emails = [
    'Hey are you okay? I want to know a playlist',
    'Hey, you have won an iPhone 14 giveaway for free. Give me your information'
]

cv_emails = cv.transform(emails)
model.predict(cv_emails)
# Output: [0, 1] → First is Ham, Second is Spam
```

---

## 📊 Results

The Multinomial Naive Bayes model was evaluated on a 20% held-out test set using accuracy score. Naive Bayes is well-known for performing exceptionally well on text classification tasks, especially spam detection, due to its strong assumption of feature independence and its ability to handle high-dimensional sparse data like word count vectors.

---

## ▶️ How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/sanzidd/Spam-Mail-Detector-ML.git
   cd Spam-Mail-Detector-ML
   ```

2. **Install required libraries**
   ```bash
   pip install numpy pandas seaborn scikit-learn
   ```

3. **Open the notebook**
   ```bash
   jupyter notebook "Spam Mail Detector ML.ipynb"
   ```

4. **Run all cells** from top to bottom.

---

## 📁 Project Structure

```
Spam-Mail-Detector-ML/
│
├── Spam Mail Detector ML.ipynb   # Main notebook with full NLP pipeline
├── emails.csv                    # Dataset
└── README.md                     # Project documentation
```

---

## 💡 Key Concepts

- **Bag of Words (CountVectorizer)** — converts raw text into a matrix of word counts, where each unique word becomes a feature
- **Multinomial Naive Bayes** — a probabilistic classifier based on Bayes' theorem, particularly suited for word count features in text classification
- **fit_transform vs transform** — `fit_transform` is used on training data to learn the vocabulary; `transform` is used on test data to apply the same vocabulary without re-learning
- **Text Classification** — the NLP task of assigning predefined categories to text documents

---

## 🙋‍♂️ Author

**Sanzid**  
[![GitHub](https://img.shields.io/badge/GitHub-sanzidd-black?style=flat&logo=github)](https://github.com/sanzidd)

---

> ⭐ If you found this project helpful, feel free to give it a star!
