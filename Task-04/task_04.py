import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------------

# 1. Load the dataset

# -----------------------------------

df = pd.read_csv(
"dataset/twitter_training.csv",
header=None
)

# Assign meaningful column names

df.columns = [
"id",
"entity",
"sentiment",
"text"
]

print("Dataset loaded successfully!")
print("Original shape:", df.shape)

# -----------------------------------

# 2. Data cleaning

# -----------------------------------

print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Remove rows with missing text or sentiment

df = df.dropna(
subset=["text", "sentiment"]
).copy()

# Convert text to string

df["text"] = df["text"].astype(str)

# Remove duplicate posts

df = df.drop_duplicates(
subset=["text", "sentiment"]
).reset_index(drop=True)

print("\nShape after cleaning:", df.shape)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# -----------------------------------

# 3. Sentiment distribution

# -----------------------------------

print("\nSentiment distribution:")
print(df["sentiment"].value_counts())

# -----------------------------------

# 4. Train-test split

# -----------------------------------

X = df["text"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42,
stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# -----------------------------------

# 5. Create TF-IDF + Logistic Regression

# -----------------------------------

model = Pipeline(
steps=[
(
"tfidf",
TfidfVectorizer(
max_features=20000,
stop_words="english",
ngram_range=(1, 2),
sublinear_tf=True
)
),
(
"classifier",
LogisticRegression(
max_iter=1000,
random_state=42
)
)
]
)

# -----------------------------------

# 6. Train model

# -----------------------------------

print("\nTraining sentiment classifier...")

model.fit(
X_train,
y_train
)

print("Model trained successfully!")

# -----------------------------------

# 7. Predictions

# -----------------------------------

y_pred = model.predict(X_test)

# -----------------------------------

# 8. Evaluation

# -----------------------------------

accuracy = accuracy_score(
y_test,
y_pred
)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Accuracy percentage: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
classification_report(
y_test,
y_pred
)
)

# -----------------------------------

# 9. Sentiment distribution chart

# -----------------------------------

plt.figure(figsize=(9, 6))

sns.countplot(
data=df,
x="sentiment",
order=df["sentiment"].value_counts().index
)

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Posts")

plt.tight_layout()

plt.savefig(
"outputs/sentiment_distribution.png",
dpi=300,
bbox_inches="tight"
)

plt.close()

# -----------------------------------

# 10. Confusion matrix

# -----------------------------------

labels = sorted(
y.unique()
)

cm = confusion_matrix(
y_test,
y_pred,
labels=labels
)

plt.figure(figsize=(8, 7))

sns.heatmap(
cm,
annot=True,
fmt="d",
cmap="Blues",
xticklabels=labels,
yticklabels=labels
)

plt.title("Sentiment Classification Confusion Matrix")
plt.xlabel("Predicted Sentiment")
plt.ylabel("Actual Sentiment")

plt.tight_layout()

plt.savefig(
"outputs/confusion_matrix.png",
dpi=300,
bbox_inches="tight"
)

plt.close()

# -----------------------------------

# 11. Finish

# -----------------------------------

print("\nTask 4 completed successfully!")

print("\nGenerated files:")
print("- outputs/sentiment_distribution.png")
print("- outputs/confusion_matrix.png")
