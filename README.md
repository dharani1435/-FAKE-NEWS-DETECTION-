# -FAKE-NEWS-DETECTION-
📌 Objective

To build a machine learning model that detects whether a news article is fake or true using Natural Language Processing (NLP).

📂 Dataset

Dataset contains two files:

Fake.csv → Fake or misleading news articles

True.csv → Verified real news articles

Each file includes: title, text, subject, date.
A new column label is added (0 = Fake, 1 = True).

⚙️ Approach

Load and merge both datasets.

Clean text: remove punctuation, numbers, URLs, and lowercase all text.

Combine title + text into one column.

Convert text to vectors using TF-IDF.

Train a Logistic Regression model.

Evaluate with accuracy, precision, recall, and F1-score.
Key Code Snippet
fake_df['label'] = 0
true_df['label'] = 1
df = pd.concat([fake_df, true_df])
df['text'] = df['title'] + " " + df['text']

tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = tfidf.fit_transform(X_train)
model = LogisticRegression(max_iter=200)
model.fit(X_train_tfidf, y_train)

Results
| Metric    | Score |
| --------- | ----- |
| Accuracy  | ~98%  |
| Precision | 0.98  |
| Recall    | 0.97  |
| F1-Score  | 0.98  |

Insights

Fake news uses emotional and exaggerated language.

True news is factual and neutral.

TF-IDF + Logistic Regression provides strong baseline performance.

🚀 Future Work

Use advanced models like BERT/RoBERTa for better context.

Add metadata (author credibility, source reputation).

Deploy as a web app via Streamlit or Flask.






















