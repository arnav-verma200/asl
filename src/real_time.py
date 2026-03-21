import cv2
import numpy as np
import joblib
import mediapipe as mp
from collections import deque
import time
import pyttsx3
import threading
import urllib.request

model      = joblib.load('models/best_model.pkl')
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands      = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Phone camera URL
SNAPSHOT_URL = "http://192.168.1.4:8080/shot.jpg"

print("Model and MediaPipe loaded!")


def get_frame(url):
    try:
        img_resp = urllib.request.urlopen(url, timeout=2)
        imgnp    = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame    = cv2.imdecode(imgnp, -1)
        return frame
    except Exception as e:
        print(f"Frame error: {e}")
        return None


def speak(text):
    def run():
        engine.say(text)
        engine.runAndWait()
    thread        = threading.Thread(target=run)
    thread.daemon = True
    thread.start()


def extract_landmarks(hand_landmarks):
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


def draw_ui(frame, predicted_letter, confidence,
            current_word, sentence, hand_detected):

    h, w = frame.shape[:2]

    font_scale_big    = w / 640
    font_scale_medium = w / 900
    font_scale_small  = w / 1200

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0),           (w, int(h*0.18)), (0, 0, 0), -1)
    cv2.rectangle(overlay, (0, int(h*0.88)), (w, h),           (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    if hand_detected:
        color  = (0, 255, 0)
        status = "Hand Detected"
    else:
        color  = (0, 0, 255)
        status = "No Hand Detected"

    cv2.putText(frame, status,
                (10, int(h*0.05)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale_medium, color, 2)

    if predicted_letter:
        cv2.putText(frame, f"Letter: {predicted_letter}",
                    (10, int(h*0.12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale_big, (0, 255, 255), 3)

        cv2.putText(frame, f"Confidence: {confidence}%",
                    (10, int(h*0.17)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale_small, (255, 255, 255), 2)

    cv2.putText(frame, f"Word: {current_word}",
                (10, int(h*0.92)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale_medium, (255, 255, 0), 2)

    cv2.putText(frame, f"Sentence: {sentence}",
                (10, int(h*0.97)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale_small, (255, 255, 255), 2)

    cv2.putText(frame, "SPACE=add word  ENTER=speak  C=clear  Q=quit",
                (int(w*0.45), int(h*0.03)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale_small, (200, 200, 200), 1)

    return frame


def main():

    prediction_buffer = deque(maxlen=10)
    current_word      = ""
    sentence          = ""
    last_prediction   = ""
    last_letter_time  = time.time()

    LETTER_DELAY      = 2.0
    CONFIDENCE_FRAMES = 7

    print(f"Connecting to: {SNAPSHOT_URL}")
    print("Testing connection...")

    test_frame = get_frame(SNAPSHOT_URL)
    if test_frame is None:
        print("Could not connect to phone camera!")
        print("Make sure:")
        print("  1. Phone and PC are on same WiFi")
        print("  2. IP Camera app is running on phone")
        print("  3. IP address is correct")
        return

    h, w = test_frame.shape[:2]
    print(f"Connected! Phone resolution: {w}x{h}")

    print("="*50)
    print("REAL TIME SIGN LANGUAGE TRANSLATOR")
    print("="*50)
    print("SPACE  →  add word to sentence")
    print("ENTER  →  speak sentence")
    print("C      →  clear everything")
    print("Q      →  quit")
    print("="*50)

    predicted_letter = ""
    confidence       = 0
    fail_count       = 0

    while True:

        frame = get_frame(SNAPSHOT_URL)

        if frame is None:
            fail_count += 1
            print(f"Failed to get frame ({fail_count})")
            if fail_count > 10:
                print("Too many failures! Check connection.")
                break
            continue

        fail_count = 0

        # Resize keeping aspect ratio
        display_width  = 1280
        h, w           = frame.shape[:2]
        display_height = int(h * display_width / w)
        frame          = cv2.resize(frame, (display_width, display_height))

        frame     = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = hands.process(image_rgb)

        hand_detected = results.multi_hand_landmarks is not None

        if hand_detected:
            for hand_lms in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0),   thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)
                )

            landmarks  = extract_landmarks(results.multi_hand_landmarks[0])
            prediction = model.predict(landmarks)[0]
            prediction_buffer.append(prediction)

            if len(prediction_buffer) == 10:
                most_common = max(set(prediction_buffer),
                                  key=list(prediction_buffer).count)
                count       = list(prediction_buffer).count(most_common)
                confidence  = round((count / 10) * 100)

                if count >= CONFIDENCE_FRAMES:
                    predicted_letter = most_common
                    current_time     = time.time()

                    if current_time - last_letter_time > LETTER_DELAY:
                        current_word    += predicted_letter
                        last_prediction  = predicted_letter
                        last_letter_time = current_time
                        print(f"Letter: {predicted_letter} | Word: {current_word}")
        else:
            prediction_buffer.clear()
            predicted_letter = ""
            confidence       = 0

        frame = draw_ui(frame, predicted_letter, confidence,
                        current_word, sentence, hand_detected)

        cv2.imshow('Sign Language Translator', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord(' '):
            if current_word:
                sentence    += current_word + " "
                speak(current_word)
                print(f"Word: {current_word} | Sentence: {sentence}")
                current_word = ""

        elif key == 13:
            if sentence:
                print(f"Speaking: {sentence}")
                speak(sentence)

        elif key == ord('c'):
            current_word = ""
            sentence     = ""
            print("Cleared!")

    cv2.destroyAllWindows()
    print("Goodbye!")


if __name__ == "__main__":
    main()