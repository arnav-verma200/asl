import cv2
import numpy as np
import joblib
import mediapipe as mp

model    = joblib.load('models/best_model.pkl')

mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)
print("Model and MediaPipe loaded!")


def extract_landmarks(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results   = hands.process(image_rgb)

    if results.multi_hand_landmarks is None:
        print("No hand detected in image!")
        return None
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
    return np.array(normalized).reshape(1, -1)


def predict_gesture(image_path):
    print(f"\nImage: {image_path}")
    landmarks = extract_landmarks(image_path)
    if landmarks is None:
        return None

    prediction       = model.predict(landmarks)
    predicted_letter = prediction[0]
    print(f"Predicted Letter: {predicted_letter}")
    return predicted_letter


if __name__ == "__main__":

    test_images = [
        'data/raw/asl_alphabet_train/asl_alphabet_train/A/A1.jpg',
        'data/raw/asl_alphabet_train/asl_alphabet_train/B/B1.jpg',
        'data/raw/asl_alphabet_train/asl_alphabet_train/C/C50.jpg',
        'data/raw/asl_alphabet_train/asl_alphabet_train/L/L1.jpg',
        'data/raw/asl_alphabet_train/asl_alphabet_train/Y/Y1.jpg',
    ]

    print("="*40)
    print("TESTING SINGLE IMAGE PREDICTIONS")
    print("="*40)

    correct = 0
    for img_path in test_images:
        true_label      = img_path.split('/')[-2]
        predicted_label = predict_gesture(img_path)

        if predicted_label == true_label:
            result = "✅ CORRECT"
            correct += 1
        else:
            result = f"❌ WRONG (true: {true_label})"

        print(f"Result: {result}")
        print("-"*40)

    print(f"\nFinal Score: {correct}/{len(test_images)} correct")