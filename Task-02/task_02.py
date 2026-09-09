import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------
# 1. Load the dataset
# -----------------------------------
df = pd.read_csv("dataset/train.csv")

print("Dataset loaded successfully!")
print("Original shape:", df.shape)

# -----------------------------------
# 2. Check missing values
# -----------------------------------
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# -----------------------------------
# 3. Data cleaning
# -----------------------------------

# Fill missing Age values with the median age
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Embarked values with the most common value
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Cabin has many missing values, so create a useful
# feature indicating whether cabin information exists
df["CabinKnown"] = df["Cabin"].notna().astype(int)

# Drop the original Cabin column because most values are missing
df = df.drop(columns=["Cabin"])

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nCleaned shape:", df.shape)

# -----------------------------------
# 4. Basic statistics
# -----------------------------------
print("\nBasic statistical summary:")
print(df.describe())

# -----------------------------------
# 5. Survival analysis
# -----------------------------------

survival_rate = df["Survived"].mean() * 100

print(f"\nOverall survival rate: {survival_rate:.2f}%")

print("\nSurvival by gender:")
print(
    df.groupby("Sex")["Survived"]
    .mean()
    .mul(100)
    .round(2)
)

print("\nSurvival by passenger class:")
print(
    df.groupby("Pclass")["Survived"]
    .mean()
    .mul(100)
    .round(2)
)

# -----------------------------------
# 6. Create output folder plots
# -----------------------------------

# Plot 1: Survival by gender
plt.figure(figsize=(8, 6))

sns.barplot(
    data=df,
    x="Sex",
    y="Survived"
)

plt.title("Survival Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Survival Rate")

plt.tight_layout()
plt.savefig(
    "outputs/survival_by_gender.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# -----------------------------------
# Plot 2: Survival by passenger class
# -----------------------------------

plt.figure(figsize=(8, 6))

sns.barplot(
    data=df,
    x="Pclass",
    y="Survived"
)

plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")

plt.tight_layout()
plt.savefig(
    "outputs/survival_by_class.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# -----------------------------------
# Plot 3: Age distribution
# -----------------------------------

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="Age",
    bins=30,
    kde=True
)

plt.title("Age Distribution of Titanic Passengers")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")

plt.tight_layout()
plt.savefig(
    "outputs/age_distribution.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# -----------------------------------
# Plot 4: Fare vs Survival
# -----------------------------------

plt.figure(figsize=(8, 6))

sns.boxplot(
    data=df,
    x="Survived",
    y="Fare"
)

plt.title("Fare Distribution by Survival")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Fare")

plt.tight_layout()
plt.savefig(
    "outputs/fare_vs_survival.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# -----------------------------------
# Plot 5: Correlation heatmap
# -----------------------------------

plt.figure(figsize=(10, 7))

numeric_columns = [
    "Survived",
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "CabinKnown"
]

correlation = df[numeric_columns].corr()

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.savefig(
    "outputs/correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# -----------------------------------
# 7. Finish
# -----------------------------------

print("\nEDA completed successfully!")

print("\nGenerated files:")
print("- outputs/survival_by_gender.png")
print("- outputs/survival_by_class.png")
print("- outputs/age_distribution.png")
print("- outputs/fare_vs_survival.png")
print("- outputs/correlation_heatmap.png")