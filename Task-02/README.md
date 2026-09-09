# Prodigy InfoTech Data Science Internship — Task 2

## Task Objective

Perform data cleaning and exploratory data analysis (EDA) on a dataset such as the Titanic dataset. Explore relationships between variables and identify patterns and trends in the data.

## Dataset

The Titanic dataset provided in the Prodigy InfoTech Data Science Internship Task 2 repository was used.

The analysis was performed using the `train.csv` dataset, which contains passenger information and the `Survived` target variable.

## Data Cleaning

The following cleaning steps were performed:

* Missing `Age` values were replaced with the median age.
* Missing `Embarked` values were replaced with the most common embarkation port.
* Because the `Cabin` column contained a large number of missing values, a new `CabinKnown` feature was created to indicate whether cabin information was available.
* The original `Cabin` column was then removed.
* Numeric variables were used to generate a correlation matrix.

## Exploratory Data Analysis

The analysis explored:

* Overall survival rate
* Survival rate by gender
* Survival rate by passenger class
* Passenger age distribution
* Fare distribution by survival status
* Correlations between numerical variables

## Key Results

### Overall Survival

The overall survival rate was **38.38%**.

### Survival by Gender

| Gender | Survival Rate |
| ------ | ------------: |
| Female |        74.20% |
| Male   |        18.89% |

Female passengers had a substantially higher survival rate than male passengers in the dataset.

### Survival by Passenger Class

| Passenger Class | Survival Rate |
| --------------: | ------------: |
|       1st Class |        62.96% |
|       2nd Class |        47.28% |
|       3rd Class |        24.24% |

Passengers in higher classes had higher survival rates than passengers in lower classes.

## Key Observations

* Gender shows a strong relationship with survival in this dataset.
* Female passengers had a much higher survival rate than male passengers.
* Passenger class also shows a clear relationship with survival.
* First-class passengers had the highest survival rate, while third-class passengers had the lowest.
* Age distribution provides insight into the passenger population and can be compared with survival outcomes.
* Fare distributions differ between passengers who survived and those who did not.
* The correlation heatmap helps identify relationships among numerical variables.

## Visualizations

The following visualizations were generated:

1. `survival_by_gender.png` — Survival rate by gender.
2. `survival_by_class.png` — Survival rate by passenger class.
3. `age_distribution.png` — Distribution of passenger ages.
4. `fare_vs_survival.png` — Fare distribution by survival status.
5. `correlation_heatmap.png` — Correlation between numerical variables.

All visualizations are stored in the `outputs` folder.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn

## Project Structure

```text
Task-02/
├── README.md
├── task_02.py
│
├── dataset/
│   ├── train.csv
│   ├── test.csv
│   └── gender_submission.csv
│
└── outputs/
    ├── survival_by_gender.png
    ├── survival_by_class.png
    ├── age_distribution.png
    ├── fare_vs_survival.png
    └── correlation_heatmap.png
```
