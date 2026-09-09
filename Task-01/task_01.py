import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load population dataset
df = pd.read_csv("dataset/population.csv", skiprows=4)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# Find latest year
year_columns = [col for col in df.columns if str(col).isdigit()]
latest_year = year_columns[-1]

print("Latest year:", latest_year)

# Convert population to numeric
df[latest_year] = pd.to_numeric(df[latest_year], errors="coerce")

# Load country metadata
metadata = pd.read_csv("dataset/metadata_country.csv")

# Keep only actual countries
country_codes = metadata[
    metadata["Region"].notna()
]["Country Code"]

# Filter population dataset using country codes
df_clean = df[
    df["Country Code"].isin(country_codes)
][["Country Name", "Country Code", latest_year]].dropna()

# Sort by population
top_10 = df_clean.sort_values(
    by=latest_year,
    ascending=False
).head(10)

print("\nTop 10 most populous countries:")
print(
    top_10[["Country Name", latest_year]].to_string(index=False)
)

# Create bar chart
plt.figure(figsize=(12, 7))

sns.barplot(
    data=top_10,
    x=latest_year,
    y="Country Name"
)

plt.title(
    f"Top 10 Most Populous Countries ({latest_year})",
    fontsize=16
)

plt.xlabel("Population")
plt.ylabel("Country")

plt.tight_layout()

# Save chart
output_path = "outputs/top_10_population.png"

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

print(f"\nChart saved successfully to: {output_path}")

