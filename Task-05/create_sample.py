import pandas as pd

source_file = r"C:\Users\Roger\.cache\kagglehub\datasets\sobhanmoosavi\us-accidents\versions\13\US_Accidents_March23.csv"

output_file = "dataset/US_Accidents_sample.csv"

target_rows = 100000

chunks = []
rows_collected = 0

print("Reading the official US Accidents dataset in chunks...")

for chunk in pd.read_csv(source_file, chunksize=25000):
    chunks.append(chunk)
    rows_collected += len(chunk)

    print("Rows collected:", rows_collected)

    if rows_collected >= target_rows:
        break

df = pd.concat(chunks, ignore_index=True).head(target_rows)

df.to_csv(output_file, index=False)

print("\nSample created successfully!")
print("Output:", output_file)
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nColumns:")
print(df.columns.tolist())