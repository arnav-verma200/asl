import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

DATASET_PATH = "data/raw/asl_alphabet_train/asl_alphabet_train"
OUTPUT_CSV   = "data/landmarks/landmarks.csv"

all_landmarks = []
all_labels    = []

for label in os.listdir(DATASET_PATH):
    label_path = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(label_path):
        continue
    print(f"Processing letter: {label}")

    for image_file in os.listdir(label_path):
        image_path = os.path.join(label_path, image_file)
        image      = cv2.imread(image_path)
        if image is None:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results   = hands.process(image_rgb)
        if results.multi_hand_landmarks is None:
            continue

        hand_landmarks = results.multi_hand_landmarks[0]
        row = []
        for landmark in hand_landmarks.landmark:
            row.append(landmark.x)
            row.append(landmark.y)

        # Normalize relative to wrist
        wrist_x = row[0]
        wrist_y = row[1]
        normalized = []
        for i in range(0, len(row), 2):
            normalized.append(row[i]   - wrist_x)
            normalized.append(row[i+1] - wrist_y)
        hand_size = max(
            abs(max(normalized[0::2]) - min(normalized[0::2])),
            abs(max(normalized[1::2]) - min(normalized[1::2]))
        )

        if hand_size > 0:
            normalized = [n / hand_size for n in normalized]
        all_landmarks.append(normalized)
        all_labels.append(label)
        
df           = pd.DataFrame(all_landmarks)
df['label']  = all_labels

os.makedirs("data/landmarks", exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)

print(f"Done! CSV saved at: {OUTPUT_CSV}")
print(f"Total rows: {len(df)}")