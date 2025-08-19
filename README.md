📰 Fake News Classifier — NLP Project 📌 Project Overview

The goal of this project is to build a text classifier that can distinguish between real news (label = 1) and fake news (label = 0).

We worked with two main datasets:

data.csv → contains training data with labels (0/1).

validation_data.csv → contains unlabeled or partially labeled data (with some labels = 2, meaning “unknown”).

The task:

- Train a classifier using the labeled dataset.
- Use the trained classifier to predict the labels for the validation dataset.
- Replace all “2” labels with the predicted labels (0 or 1) while preserving the file format.

⚙️ Steps in the Pipeline

1. Data Loading & Exploration
- Checked dataset sizes.
- Verified missing values (missing ratio).
- Removed duplicate rows to avoid data leakage.
1. Text Preprocessing
- We cleaned and standardized the text before feeding it to models:
- Lowercasing → unify words.
- Removing stopwords → remove uninformative words like “the”, “and”.
- Stemming → cut words to their root form (e.g., playing → play).
- Lemmatization → bring words to their base dictionary form (e.g., better → good).
- Saved cleaned data into train_ready.csv and validation_ready.csv.
1. Feature Extraction (Vectorization)
- Used TF-IDF (Term Frequency – Inverse Document Frequency):
- Captures both frequency of words and how informative they are across documents.
- Better than Bag of Words alone, because it reduces the importance of very common words.

Parameters:

ngram_range=(1,2) → includes both unigrams (single words) and bigrams (pairs of words).

min_df=2 → keep words appearing in at least 2 documents.

max_df=0.9 → discard words appearing in more than 90% of documents.

max_features=50000 → limit vocabulary size.

1. Model Training
- We trained and compared two classifiers:
- Naive Bayes (with Bag of Words / TF-IDF embeddings).
- Support Vector Machine (SVM) — turned out to perform better.
1. Evaluation Metrics

We used:

- Accuracy → overall % of correct predictions.
- Precision → how many predicted “fake news” were actually fake.
- Recall → how many actual fake news were correctly identified.
- F1-score → harmonic mean of precision & recall.

We also plotted confusion matrices (both raw counts and normalized percentages) to visualize model performance.

📊 Example interpretation:

High precision → fewer false alarms (good at not labeling real news as fake).

High recall → fewer misses (good at catching fake news).

1. Validation Predictions
- Implemented a helper pipeline to:
- Load the trained model and vectorizer (joblib).
- Clean text (with preprocessing pipeline or fallback cleaner).
- Run predictions on validation dataset.
- Replace only rows where label = 2 with predictions.
- Save results in the original format.

Output file → validation_predictions.csv

🏆 Results

SVM outperformed Naive Bayes in Accuracy, Precision, Recall, and F1-score.

This means SVM is better at handling subtle patterns in text and distinguishing fake from real news.

🚀 How to Run

Preprocess the data Run the preprocessing notebook to generate train_ready.csv and validation_ready.csv.

Train the model Choose your feature extractor (TF-IDF) and model (Naive Bayes or SVM).

Evaluate the model Use confusion matrix and metrics to assess performance.

Predict on validation dataset Run predict_and_replace_on_validation() → saves a new CSV with predictions filled in.

✨ Key Learnings

Text preprocessing (cleaning, lemmatization, stopword removal) is critical.

TF-IDF captures text meaning better than plain Bag of Words.

SVM outperforms Naive Bayes on this dataset.

Confusion matrices help visualize errors better than accuracy alone.
