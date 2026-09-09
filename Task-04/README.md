# Prodigy InfoTech Data Science Internship — Task 4

## Task Objective

Perform sentiment analysis on social media posts to identify whether the expressed sentiment is positive, negative, neutral, or irrelevant.

## Dataset

The dataset used for this task is the **Twitter Entity Sentiment Analysis** dataset provided through the Prodigy InfoTech Data Science Internship dataset repository.

The dataset contains social media posts associated with different entities and their corresponding sentiment labels.

The four sentiment categories are:

* **Positive**
* **Negative**
* **Neutral**
* **Irrelevant**

## Dataset Structure

The original dataset contains four fields:

| Column    | Description                           |
| --------- | ------------------------------------- |
| ID        | Identifier associated with the post   |
| Entity    | Entity/topic associated with the post |
| Sentiment | Sentiment label                       |
| Text      | Social media post text                |

The dataset contains approximately **74,682 records** before cleaning.

## Data Cleaning

The following preprocessing steps were performed:

1. Assigned meaningful column names to the dataset.
2. Removed records with missing text or sentiment values.
3. Converted text values to strings.
4. Removed duplicate posts with identical text and sentiment labels.
5. Used the cleaned text as the input feature and sentiment as the target.

## Sentiment Distribution

The original sentiment distribution was:

| Sentiment  | Number of Posts |
| ---------- | --------------: |
| Negative   |          22,542 |
| Positive   |          20,832 |
| Neutral    |          18,318 |
| Irrelevant |          12,990 |

This distribution was visualized using a bar chart.

## Machine Learning Approach

A lightweight text-classification pipeline was developed using:

### TF-IDF Vectorization

**TF-IDF (Term Frequency–Inverse Document Frequency)** was used to convert text into numerical features.

The configuration included:

* Maximum features: 20,000
* English stop-word removal
* Unigrams and bigrams
* Sublinear TF scaling

### Logistic Regression

A **Logistic Regression** classifier was trained on the TF-IDF features to classify posts into the four sentiment categories.

The dataset was divided using an **80/20 stratified train-test split**.

## Model Evaluation

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

### Results

The model achieved approximately:

* **Accuracy: 79%**
* **Macro F1-score: 0.78**
* **Weighted F1-score: 0.79**

The test dataset contained **13,954 samples**.

## Visualizations

### 1. Sentiment Distribution

`outputs/sentiment_distribution.png`

Shows the number of posts belonging to each sentiment category.

### 2. Confusion Matrix

`outputs/confusion_matrix.png`

Shows the model's classification performance across the four sentiment categories.

## Key Observations

* The dataset contains four sentiment classes.
* Negative sentiment had the largest number of samples.
* Irrelevant sentiment had the smallest number of samples.
* TF-IDF provided an efficient way to represent the text numerically.
* Logistic Regression achieved approximately 79% accuracy on the test set.
* The confusion matrix provides insight into which sentiment categories are more difficult for the model to distinguish.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

## Project Structure

```text
Task-04/
├── README.md
├── task_04.py
│
├── dataset/
│   └── twitter_training.csv
│
└── outputs/
    ├── sentiment_distribution.png
    └── confusion_matrix.png
```

## Conclusion

A machine learning-based sentiment analysis system was successfully developed using TF-IDF and Logistic Regression. The model achieved approximately 79% accuracy while classifying social media posts into positive, negative, neutral, and irrelevant sentiment categories.
