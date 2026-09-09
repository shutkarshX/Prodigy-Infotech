# Prodigy InfoTech Data Science Internship — Task 1

## Task Objective

Create a bar chart or histogram to visualize the distribution of a categorical or continuous variable, such as the distribution of ages or genders in a population.

## Dataset

The dataset used for this task is the **Population, total** dataset provided in the Prodigy InfoTech Data Science Internship Task 1 repository.

The dataset contains population data for countries and regions across multiple years.

For this analysis, the latest available year in the dataset, **2024**, was used.

## Approach

The following steps were performed:

1. Loaded the population dataset using Pandas.
2. Identified the latest available year automatically.
3. Converted population values to numeric format.
4. Used the country metadata to distinguish actual countries from regional and income-group aggregates.
5. Sorted countries according to their 2024 population.
6. Selected the top 10 most populous countries.
7. Created a horizontal bar chart using Matplotlib and Seaborn.
8. Saved the visualization as a PNG file in the `outputs` folder.

## Top 10 Most Populous Countries — 2024

| Rank | Country            |    Population |
| ---: | ------------------ | ------------: |
|    1 | India              | 1,450,935,791 |
|    2 | China              | 1,408,975,000 |
|    3 | United States      |   340,110,988 |
|    4 | Indonesia          |   283,487,931 |
|    5 | Pakistan           |   251,269,164 |
|    6 | Nigeria            |   232,679,478 |
|    7 | Brazil             |   211,998,573 |
|    8 | Bangladesh         |   173,562,364 |
|    9 | Russian Federation |   143,533,851 |
|   10 | Ethiopia           |   132,059,767 |

## Key Observations

* India has the highest population among the countries included in the dataset for 2024.
* China has the second-highest population.
* India and China have populations substantially larger than the other countries in the top 10.
* The United States ranks third, followed by Indonesia and Pakistan.
* Six of the top 10 countries are from Asia.
* Nigeria and Ethiopia are the two African countries appearing in the top 10.
* The bar chart makes the difference in population between the countries easy to compare visually.

## Output

The generated visualization is saved as:

```text
outputs/top_10_population.png
```

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn

## Project Structure

```text
Task-01/
├── dataset/
│   ├── population.csv
│   └── metadata_country.csv
├── outputs/
│   └── top_10_population.png
├── task_01.py
└── README.md
```
