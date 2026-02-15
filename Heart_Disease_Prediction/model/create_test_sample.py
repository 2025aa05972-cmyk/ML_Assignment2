import pandas as pd
import os

# Load full dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
df = pd.read_csv(os.path.join(project_root, "dataset", "heart_disease_uci.csv"))

# Drop rows with missing target
df = df.dropna(subset=["num"])

# Take random sample (200 rows)
test_sample = df.sample(n=200, random_state=42)

# Save to project root
test_sample.to_csv(os.path.join(project_root, "dataset", "test_sample.csv"), index=False)

print("test_sample.csv created successfully!")
