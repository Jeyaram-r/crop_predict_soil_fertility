import pandas as pd

# Load the dataset
file_path = "crop_recommendation.csv"
df = pd.read_csv(file_path)

# Get the unique crop names
unique_crops = df['label'].unique()
print(unique_crops)
