import pandas as pd

# Read the data
print("Reading IRON.csv...")
iron_df = pd.read_csv('dataset/IRON/IRON.csv')

print("Reading IRON_processed.csv...")
processed_df = pd.read_csv('dataset/IRON/IRON_processed.csv')

# Add date column
print("Adding date column...")
processed_df.insert(0, 'date', iron_df['date'])

# Save the result
print("Saving to IRON_processed_with_date.csv...")
processed_df.to_csv('dataset/IRON/IRON_processed_with_date.csv', index=False)

print("Done!")
print(f"Shape: {processed_df.shape}")
print("First 5 rows:")
print(processed_df.head())
print("Columns:", list(processed_df.columns))
