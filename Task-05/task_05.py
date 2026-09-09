import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------

# 1. Load the dataset

# -----------------------------------

df = pd.read_csv(
"dataset/US_Accidents_sample.csv"
)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# -----------------------------------

# 2. Data preparation

# -----------------------------------

# Convert accident start time to datetime

df["Start_Time"] = pd.to_datetime(
df["Start_Time"],
errors="coerce"
)

# Remove rows with invalid timestamps

df = df.dropna(
subset=["Start_Time"]
).copy()

# Extract time-related features

df["Hour"] = df["Start_Time"].dt.hour
df["Day_of_Week"] = df["Start_Time"].dt.day_name()
df["Month"] = df["Start_Time"].dt.month_name()

# Create time-of-day categories

def get_time_of_day(hour):
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"

df["Time_of_Day"] = df["Hour"].apply(
get_time_of_day
)

# -----------------------------------

# 3. Basic statistics

# -----------------------------------

print("\nMissing values:")
print(
df.isnull().sum()
.sort_values(ascending=False)
.head(15)
)

print("\nSeverity distribution:")
print(
df["Severity"]
.value_counts()
.sort_index()
)

# -----------------------------------

# 4. Accident patterns by time of day

# -----------------------------------

time_order = [
"Morning",
"Afternoon",
"Evening",
"Night"
]

time_counts = (
df["Time_of_Day"]
.value_counts()
.reindex(time_order)
)

print("\nAccidents by time of day:")
print(time_counts)

plt.figure(figsize=(9, 6))

sns.barplot(
x=time_counts.index,
y=time_counts.values
)

plt.title("Accidents by Time of Day")
plt.xlabel("Time of Day")
plt.ylabel("Number of Accidents")

plt.tight_layout()

plt.savefig(
"outputs/accidents_by_time_of_day.png",
dpi=300,
bbox_inches="tight"
)

plt.close()

# -----------------------------------

# 5. Accidents by weather condition

# -----------------------------------

weather_counts = (
df["Weather_Condition"]
.fillna("Unknown")
.value_counts()
.head(10)
)

print("\nTop 10 weather conditions:")
print(weather_counts)

plt.figure(figsize=(10, 7))

sns.barplot(
x=weather_counts.values,
y=weather_counts.index
)

plt.title("Top 10 Weather Conditions During Accidents")
plt.xlabel("Number of Accidents")
plt.ylabel("Weather Condition")

plt.tight_layout()

plt.savefig(
"outputs/accidents_by_weather.png",
dpi=300,
bbox_inches="tight"
)

plt.close()

# -----------------------------------

# 6. Accidents by state

# -----------------------------------

state_counts = (
df["State"]
.value_counts()
.head(15)
)

print("\nTop 15 states by accident count:")
print(state_counts)

plt.figure(figsize=(10, 7))

sns.barplot(
x=state_counts.values,
y=state_counts.index
)

plt.title("Top 15 States by Accident Count")
plt.xlabel("Number of Accidents")
plt.ylabel("State")

plt.tight_layout()

plt.savefig(
"outputs/top_states.png",
dpi=300,
bbox_inches="tight"
)

plt.close()

# -----------------------------------

# 7. Accident hotspots by city

# -----------------------------------

city_counts = (
df["City"]
.fillna("Unknown")
.value_counts()
.head(15)
)

print("\nTop 15 accident hotspot cities:")
print(city_counts)

plt.figure(figsize=(10, 7))

sns.barplot(
x=city_counts.values,
y=city_counts.index
)

plt.title("Top 15 Accident Hotspot Cities")
plt.xlabel("Number of Accidents")
plt.ylabel("City")

plt.tight_layout()

plt.savefig(
"outputs/accident_hotspots.png",
dpi=300,
bbox_inches="tight"
)

plt.close()

# -----------------------------------

# 8. Road-related contributing factors

# -----------------------------------

road_factors = [
"Amenity",
"Bump",
"Crossing",
"Give_Way",
"Junction",
"No_Exit",
"Railway",
"Roundabout",
"Station",
"Stop",
"Traffic_Calming",
"Traffic_Signal"
]

factor_counts = {}

for factor in road_factors:
    if factor in df.columns:
        factor_counts[factor] = int(
            df[factor].fillna(False).astype(bool).sum()
        )

factor_df = (
pd.Series(factor_counts)
.sort_values(ascending=False)
)

print("\nRoad-related factors:")
print(factor_df)

plt.figure(figsize=(10, 7))

sns.barplot(
x=factor_df.values,
y=factor_df.index
)

plt.title("Road-Related Factors Associated With Accidents")
plt.xlabel("Number of Records")
plt.ylabel("Road Feature")

plt.tight_layout()

plt.savefig(
"outputs/road_factors.png",
dpi=300,
bbox_inches="tight"
)

plt.close()

# -----------------------------------

# 9. Severity by time of day

# -----------------------------------

severity_time = pd.crosstab(
df["Time_of_Day"],
df["Severity"]
)

severity_time = severity_time.reindex(
time_order
)

print("\nSeverity by time of day:")
print(severity_time)

plt.figure(figsize=(10, 7))

severity_time.plot(
kind="bar",
figsize=(10, 7)
)

plt.title("Accident Severity by Time of Day")
plt.xlabel("Time of Day")
plt.ylabel("Number of Accidents")
plt.xticks(rotation=0)
plt.legend(title="Severity")

plt.tight_layout()

plt.savefig(
"outputs/severity_by_time.png",
dpi=300,
bbox_inches="tight"
)

plt.close()

# -----------------------------------

# 10. Geographic hotspot map

# -----------------------------------

# Use a smaller subset for plotting

# to keep the visualization lightweight.

map_data = df[
["Start_Lat", "Start_Lng", "Severity"]
].dropna()

map_data = map_data.sample(
n=min(10000, len(map_data)),
random_state=42
)

plt.figure(figsize=(10, 7))

scatter = plt.scatter(
map_data["Start_Lng"],
map_data["Start_Lat"],
c=map_data["Severity"],
alpha=0.35,
s=8
)

plt.title("Geographic Distribution of Accident Records")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.colorbar(
scatter,
label="Severity"
)

plt.tight_layout()

plt.savefig(
"outputs/geographic_hotspots.png",
dpi=300,
bbox_inches="tight"
)

plt.close()

# -----------------------------------

# 11. Finish

# -----------------------------------

print("\nTask 5 completed successfully!")

print("\nGenerated files:")
print("- outputs/accidents_by_time_of_day.png")
print("- outputs/accidents_by_weather.png")
print("- outputs/top_states.png")
print("- outputs/accident_hotspots.png")
print("- outputs/road_factors.png")
print("- outputs/severity_by_time.png")
print("- outputs/geographic_hotspots.png")
