import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------------

# 1. Load the dataset

# -----------------------------------

df = pd.read_csv(
"dataset/bank-additional-full.csv",
sep=";"
)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# -----------------------------------

# 2. Basic data inspection

# -----------------------------------

print("\nTarget distribution:")
print(df["y"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())

# -----------------------------------

# 3. Prepare features and target

# -----------------------------------

X = df.drop(columns=["y"])
y = df["y"].map({
"no": 0,
"yes": 1
})

# Identify categorical and numerical columns

categorical_columns = X.select_dtypes(
include=["object"]
).columns.tolist()

numerical_columns = X.select_dtypes(
exclude=["object"]
).columns.tolist()

print("\nCategorical columns:")
print(categorical_columns)

print("\nNumerical columns:")
print(numerical_columns)

# -----------------------------------

# 4. Preprocessing

# -----------------------------------

preprocessor = ColumnTransformer(
transformers=[
(
"categorical",
OneHotEncoder(
handle_unknown="ignore",
sparse_output=False
),
categorical_columns
),
(
"numerical",
"passthrough",
numerical_columns
)
]
)

# -----------------------------------

# 5. Create Decision Tree model

# -----------------------------------

model = DecisionTreeClassifier(
max_depth=6,
random_state=42,
class_weight="balanced"
)

pipeline = Pipeline(
steps=[
("preprocessor", preprocessor),
("classifier", model)
]
)

# -----------------------------------

# 6. Split the dataset

# -----------------------------------

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

# 7. Train the model

# -----------------------------------

print("\nTraining Decision Tree...")

pipeline.fit(
X_train,
y_train
)

print("Model trained successfully!")

# -----------------------------------

# 8. Make predictions

# -----------------------------------

y_pred = pipeline.predict(X_test)

# -----------------------------------

# 9. Evaluate the model

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
y_pred,
target_names=["No", "Yes"]
)
)

# -----------------------------------

# 10. Confusion Matrix

# -----------------------------------

cm = confusion_matrix(
y_test,
y_pred
)

plt.figure(figsize=(7, 6))

sns.heatmap(
cm,
annot=True,
fmt="d",
cmap="Blues",
xticklabels=["No", "Yes"],
yticklabels=["No", "Yes"]
)

plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
"outputs/confusion_matrix.png",
dpi=300,
bbox_inches="tight"
)

plt.close()

# -----------------------------------

# 11. Visualize the Decision Tree

# -----------------------------------

# Transform the training data so that the tree

# can be visualized with feature names.

X_train_transformed = pipeline.named_steps[
"preprocessor"
].transform(X_train)

feature_names = pipeline.named_steps[
"preprocessor"
].get_feature_names_out()

plt.figure(figsize=(24, 12))

plot_tree(
pipeline.named_steps["classifier"],
feature_names=feature_names,
class_names=["No", "Yes"],
filled=True,
rounded=True,
max_depth=3,
fontsize=8
)

plt.title("Decision Tree Classifier")

plt.tight_layout()

plt.savefig(
"outputs/decision_tree.png",
dpi=300,
bbox_inches="tight"
)

plt.close()

# -----------------------------------

# 12. Feature importance

# -----------------------------------

importances = pipeline.named_steps[
"classifier"
].feature_importances_

feature_importance_df = pd.DataFrame({
"Feature": feature_names,
"Importance": importances
})

feature_importance_df = feature_importance_df.sort_values(
by="Importance",
ascending=False
).head(15)

plt.figure(figsize=(10, 7))

sns.barplot(
data=feature_importance_df,
x="Importance",
y="Feature"
)

plt.title("Top 15 Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")

plt.tight_layout()

plt.savefig(
"outputs/feature_importance.png",
dpi=300,
bbox_inches="tight"
)

plt.close()

# -----------------------------------

# 13. Finish

# -----------------------------------

print("\nTask 3 completed successfully!")

print("\nGenerated files:")
print("- outputs/confusion_matrix.png")
print("- outputs/decision_tree.png")
print("- outputs/feature_importance.png")
