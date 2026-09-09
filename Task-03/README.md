# Prodigy InfoTech Data Science Internship — Task 3

## Task Objective

Build a decision tree classifier to predict whether a customer will purchase a product or service based on their demographic and behavioral data.

The Bank Marketing dataset was used for this task to predict whether a customer subscribed to a term deposit.

## Dataset

The dataset used is the **Bank Marketing dataset**.

It contains demographic, campaign-related, and economic information about bank customers.

The target variable is:

* `y = yes` — customer subscribed to the term deposit
* `y = no` — customer did not subscribe

The dataset contains **41,188 records and 21 columns**.

## Features Used

The dataset contains both numerical and categorical features.

### Categorical Features

* `job`
* `marital`
* `education`
* `default`
* `housing`
* `loan`
* `contact`
* `month`
* `day_of_week`
* `poutcome`

### Numerical Features

* `age`
* `duration`
* `campaign`
* `pdays`
* `previous`
* `emp.var.rate`
* `cons.price.idx`
* `cons.conf.idx`
* `euribor3m`
* `nr.employed`

## Data Preprocessing

The following preprocessing steps were performed:

1. Separated the target variable `y` from the input features.
2. Converted the target:

   * `no` → `0`
   * `yes` → `1`
3. Identified categorical and numerical features automatically.
4. Applied **One-Hot Encoding** to categorical features.
5. Kept numerical features unchanged.
6. Used an 80/20 stratified train-test split.

## Model

A **Decision Tree Classifier** from Scikit-learn was used.

Model configuration:

* Maximum tree depth: `6`
* Random state: `42`
* Class weighting: `balanced`

Class balancing was used because the target classes are imbalanced.

## Model Evaluation

The trained model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

The model achieved approximately **85% accuracy** on the test dataset.

The test set contained **8,238 samples**.

## Visualizations

Three visualizations were generated:

### 1. Confusion Matrix

`outputs/confusion_matrix.png`

Shows the number of correct and incorrect predictions for each class.

### 2. Decision Tree

`outputs/decision_tree.png`

Visualizes the structure of the trained Decision Tree classifier.

### 3. Feature Importance

`outputs/feature_importance.png`

Shows the most influential features used by the Decision Tree when making predictions.

## Key Observations

* The Decision Tree achieved approximately 85% accuracy.
* Both demographic and behavioral information were used for prediction.
* One-hot encoding allowed categorical customer information to be processed by the machine learning model.
* Feature importance provides insight into which variables contributed most to the model's predictions.
* The confusion matrix provides a detailed view of classification performance for subscribed and non-subscribed customers.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

## Project Structure

```text
Task-03/
├── README.md
├── task_03.py
│
├── dataset/
│   └── bank-additional-full.csv
│
└── outputs/
    ├── confusion_matrix.png
    ├── decision_tree.png
    └── feature_importance.png
```

## Conclusion

A Decision Tree classification model was successfully developed to predict whether bank customers would subscribe to a term deposit. The model achieved approximately 85% test accuracy, while the generated visualizations provide additional insight into model performance and feature importance.
