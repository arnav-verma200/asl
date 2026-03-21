import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# Load CSV
df = pd.read_csv('data/landmarks/landmarks.csv')
print(f"Original shape:  {df.shape}")
print(f"Classes found:   {df['label'].unique()}")

# Remove useless classes
classes_to_remove = ['nothing', 'del']
df = df[~df['label'].isin(classes_to_remove)]
print(f"After removing useless classes: {df.shape}")

# Balance classes
min_count = df['label'].value_counts().min()
print(f"Minimum class count: {min_count}")

# New approach - loop through each label manually
balanced_dfs = []

for label in df['label'].unique():
    label_df = df[df['label'] == label]
    sampled_df = label_df.sample(min_count, random_state=42)
    balanced_dfs.append(sampled_df)

df_balanced = pd.concat(balanced_dfs).reset_index(drop=True)

print(f"After balancing: {df_balanced.shape}")
print(df_balanced['label'].value_counts())

# Save clean CSV
df_balanced.to_csv('data/landmarks/landmarks_clean.csv', index=False)
print("Saved landmarks_clean.csv")

# Separate features and label
X = df_balanced.drop('label', axis=1).values
y = df_balanced['label'].values
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# First split - test set
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y,
    test_size=0.15,
    random_state=42,
    stratify=y
)

# Second split - validation set
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val,
    test_size=0.176,
    random_state=42,
    stratify=y_train_val
)

print(f"Training set:    {X_train.shape}")
print(f"Validation set:  {X_val.shape}")
print(f"Test set:        {X_test.shape}")

# Save all splits
os.makedirs('data/processed', exist_ok=True)

np.save('data/processed/X_train.npy', X_train)
np.save('data/processed/X_val.npy',   X_val)
np.save('data/processed/X_test.npy',  X_test)
np.save('data/processed/y_train.npy', y_train)
np.save('data/processed/y_val.npy',   y_val)
np.save('data/processed/y_test.npy',  y_test)

print("All splits saved to data/processed/")
print("Preprocessing complete!")