import zipfile
import os

# Path of your downloaded ZIP dataset
zip_path = r"C:\Users\Vandana M\OneDrive\Documents\Downloads\images for phrases.zip"

# Dataset extraction folder
extract_path = "data/dataset"

# Create folder if it doesn't exist
os.makedirs(extract_path, exist_ok=True)

# Extract ZIP dataset
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Dataset extracted successfully!")
print("Dataset location:", extract_path)