import cv2
import numpy as np
import joblib
import mediapipe as mp
from collections import deque
import time
import pyttsx3
import threading
import urllib.request

# ─────────────────────────────
# GLOBAL VARIABLES
# ─────────────────────────────
pinky_hold_start  = None
motion_trajectory = []
motion_mode       = None
HOLD_TIME         = 2.0
MAX_TRAJECTORY    = 60

model      = joblib.load('models/best_model.pkl')
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands      = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

speak_lock   = threading.Lock()
SNAPSHOT_URL = "http://192.168.1.4:8080/shot.jpg"

print("Model and MediaPipe loaded!")


# ─────────────────────────────
# CAMERA
# ─────────────────────────────
def get_frame(url):
    try:
        img_resp = urllib.request.urlopen(url, timeout=2)
        imgnp    = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        return cv2.imdecode(imgnp, -1)
    except Exception as e:
        print(f"Frame error: {e}")
        return None


# ─────────────────────────────
# SPEECH
# ─────────────────────────────
def speak(text):
    def run():
        with speak_lock:
            try:
                tts = pyttsx3.init()
                tts.setProperty('rate', 150)
                tts.say(text)
                tts.runAndWait()
                tts.stop()
            except Exception as e:
                print(f"Speech error: {e}")
    t        = threading.Thread(target=run)
    t.daemon = True
    t.start()


# ─────────────────────────────
# LANDMARK NORMALIZATION
# ─────────────────────────────
def extract_landmarks(hand_landmarks):
    row = []
    for landmark in hand_landmarks.landmark:
        row.append(landmark.x)
        row.append(landmark.y)

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


# ─────────────────────────────
# FINGER STATE DETECTION
# ─────────────────────────────
def is_finger_up(landmarks, tip_id, base_id):
    return landmarks[tip_id].y < landmarks[base_id].y


def is_pinky_only_up(hand_landmarks):
    lm = hand_landmarks.landmark
    return (
        is_finger_up(lm, 20, 17) and      # pinky up
        not is_finger_up(lm, 16, 13) and  # ring down
        not is_finger_up(lm, 12, 9)  and  # middle down
        not is_finger_up(lm, 8,  5)       # index down
    )


def is_pointer_only_up(hand_landmarks):
    lm = hand_landmarks.landmark
    return (
        is_finger_up(lm, 8,  5)  and      # index up
        not is_finger_up(lm, 12, 9)  and  # middle down
        not is_finger_up(lm, 16, 13) and  # ring down
        not is_finger_up(lm, 20, 17)      # pinky down
    )


# ─────────────────────────────
# MOVEMENT DETECTION
# ─────────────────────────────
def is_hand_still(trajectory, threshold=0.04):
    if len(trajectory) < 10:
        return True
    xs = [p[0] for p in trajectory]
    ys = [p[1] for p in trajectory]
    return (max(xs) - min(xs)) < threshold and \
           (max(ys) - min(ys)) < threshold


# ─────────────────────────────
# J SHAPE DETECTION
# ─────────────────────────────
def is_J_shape(trajectory, min_distance=0.1):
    if len(trajectory) < 20:
        return False

    xs = [p[0] for p in trajectory]
    ys = [p[1] for p in trajectory]

    dy  = ys[-1] - ys[0]   # positive = moved DOWN
    dx  = xs[-1] - xs[0]   # negative = moved LEFT

    mid            = len(xs) // 2
    second_half_dx = xs[-1] - xs[mid]

    moved_down     = dy > min_distance
    curved_left    = dx < 0 or second_half_dx < -0.02
    significant    = (abs(dy) + abs(dx)) > min_distance

    return moved_down and curved_left and significant


# ─────────────────────────────
# Z SHAPE DETECTION
# ─────────────────────────────
def is_Z_shape(trajectory, min_distance=0.08):
    if len(trajectory) < 20:
        return False

    n  = len(trajectory)
    s1 = trajectory[:n//3]
    s2 = trajectory[n//3:2*n//3]
    s3 = trajectory[2*n//3:]

    dx1 = s1[-1][0] - s1[0][0]  # segment 1 → right
    dx2 = s2[-1][0] - s2[0][0]  # segment 2 → left
    dx3 = s3[-1][0] - s3[0][0]  # segment 3 → right

    all_xs    = [p[0] for p in trajectory]
    significant = (max(all_xs) - min(all_xs)) > min_distance

    return (dx1 > 0.02 and
            dx2 < -0.02 and
            dx3 > 0.02 and
            significant)


# ─────────────────────────────
# J AND Z CONTROLLER
# ─────────────────────────────
def check_J_Z(hand_landmarks, current_time):
    global pinky_hold_start, motion_trajectory, motion_mode

    lm        = hand_landmarks.landmark
    pinky_tip = (lm[20].x, lm[20].y)
    index_tip = (lm[8].x,  lm[8].y)

    # ── J LOGIC ──
    if is_pinky_only_up(hand_landmarks):

        if motion_mode == 'J':
            motion_trajectory.append(pinky_tip)
            if len(motion_trajectory) > MAX_TRAJECTORY:
                motion_trajectory.pop(0)
            if is_J_shape(motion_trajectory):
                motion_trajectory = []
                motion_mode       = None
                pinky_hold_start  = None
                return 'J'

        elif not is_hand_still(motion_trajectory):
            motion_mode = 'J'
            motion_trajectory.append(pinky_tip)

        else:
            if pinky_hold_start is None:
                pinky_hold_start  = current_time
                motion_trajectory = [pinky_tip]
            else:
                motion_trajectory.append(pinky_tip)
                if current_time - pinky_hold_start >= HOLD_TIME:
                    pinky_hold_start  = None
                    motion_trajectory = []
                    motion_mode       = None
                    return 'I'

    # ── Z LOGIC ──
    elif is_pointer_only_up(hand_landmarks):

        if motion_mode == 'Z':
            motion_trajectory.append(index_tip)
            if len(motion_trajectory) > MAX_TRAJECTORY:
                motion_trajectory.pop(0)
            if is_Z_shape(motion_trajectory):
                motion_trajectory = []
                motion_mode       = None
                return 'Z'

        elif not is_hand_still(motion_trajectory):
            motion_mode = 'Z'
            motion_trajectory.append(index_tip)

        else:
            motion_trajectory = [index_tip]
            motion_mode       = None

    # ── RESET ──
    else:
        pinky_hold_start  = None
        motion_trajectory = []
        motion_mode       = None

    return None


# ─────────────────────────────
# UI
# ─────────────────────────────
def draw_ui(frame, predicted_letter, confidence,
            current_word, sentence, hand_detected):

    h, w = frame.shape[:2]

    font_big    = w / 640
    font_medium = w / 900
    font_small  = w / 1200

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0),           (w, int(h*0.18)), (0, 0, 0), -1)
    cv2.rectangle(overlay, (0, int(h*0.88)), (w, h),           (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Hand status
    color  = (0, 255, 0)   if hand_detected else (0, 0, 255)
    status = "Hand Detected" if hand_detected else "No Hand Detected"
    cv2.putText(frame, status, (10, int(h*0.05)),
                cv2.FONT_HERSHEY_SIMPLEX, font_medium, color, 2)

    # Motion mode indicator
    if motion_mode:
        cv2.putText(frame, f"Motion Mode: {motion_mode}",
                    (10, int(h*0.085)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_small, (0, 255, 255), 2)

    # J hold progress indicator
    if pinky_hold_start is not None and motion_mode is None:
        held    = time.time() - pinky_hold_start
        progress = min(held / HOLD_TIME, 1.0)
        bar_w   = int(w * 0.3 * progress)
        cv2.rectangle(frame,
                      (10, int(h*0.09)),
                      (10 + int(w*0.3), int(h*0.10)),
                      (50, 50, 50), -1)
        cv2.rectangle(frame,
                      (10, int(h*0.09)),
                      (10 + bar_w, int(h*0.10)),
                      (0, 255, 255), -1)
        cv2.putText(frame, "Hold for I...",
                    (10, int(h*0.085)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_small, (0, 255, 255), 1)

    # Predicted letter
    if predicted_letter:
        cv2.putText(frame, f"Letter: {predicted_letter}",
                    (10, int(h*0.12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_big, (0, 255, 255), 3)
        cv2.putText(frame, f"Confidence: {confidence}%",
                    (10, int(h*0.17)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_small, (255, 255, 255), 2)

    # Word and sentence
    cv2.putText(frame, f"Word: {current_word}",
                (10, int(h*0.92)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_medium, (255, 255, 0), 2)
    cv2.putText(frame, f"Sentence: {sentence}",
                (10, int(h*0.97)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_small, (255, 255, 255), 2)

    # Controls
    cv2.putText(frame, "SPACE=add word  ENTER=speak  C=clear  Q=quit",
                (int(w*0.45), int(h*0.03)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_small, (200, 200, 200), 1)

    return frame


# ─────────────────────────────
# MAIN
# ─────────────────────────────
def main():

    prediction_buffer = deque(maxlen=10)
    current_word      = ""
    sentence          = ""
    last_prediction   = ""
    last_letter_time  = time.time()

    LETTER_DELAY      = 2.0
    CONFIDENCE_FRAMES = 7

    print(f"Connecting to: {SNAPSHOT_URL}")
    test_frame = get_frame(SNAPSHOT_URL)

    if test_frame is None:
        print("Could not connect! Check IP and WiFi.")
        return

    h, w = test_frame.shape[:2]
    print(f"Connected! Resolution: {w}x{h}")
    print("="*50)
    print("REAL TIME SIGN LANGUAGE TRANSLATOR")
    print("="*50)
    print("SPACE  →  add word")
    print("ENTER  →  speak")
    print("C      →  clear")
    print("Q      →  quit")
    print("="*50)

    predicted_letter = ""
    confidence       = 0
    fail_count       = 0

    while True:

        frame = get_frame(SNAPSHOT_URL)

        if frame is None:
            fail_count += 1
            if fail_count > 10:
                print("Connection lost!")
                break
            continue

        fail_count = 0

        display_width  = 1280
        h, w           = frame.shape[:2]
        display_height = int(h * display_width / w)
        frame          = cv2.resize(frame, (display_width, display_height))
        frame          = cv2.flip(frame, 1)

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

            hand_lm      = results.multi_hand_landmarks[0]
            current_time = time.time()

            # ── Check J or Z first ──
            special = check_J_Z(hand_lm, current_time)

            if special:
                predicted_letter = special
                confidence       = 100
                prediction_buffer.clear()

                if current_time - last_letter_time > LETTER_DELAY:
                    current_word    += special
                    last_prediction  = special
                    last_letter_time = current_time
                    print(f"Letter: {special} | Word: {current_word}")

            else:
                # ── Normal SVM for everything else ──
                landmarks  = extract_landmarks(hand_lm)
                prediction = model.predict(landmarks)[0]
                prediction_buffer.append(prediction)

                if len(prediction_buffer) == 10:
                    most_common = max(set(prediction_buffer),
                                      key=list(prediction_buffer).count)
                    count       = list(prediction_buffer).count(most_common)
                    confidence  = round((count / 10) * 100)

                    if count >= CONFIDENCE_FRAMES:
                        predicted_letter = most_common
                        if current_time - last_letter_time > LETTER_DELAY:
                            current_word    += most_common
                            last_prediction  = most_common
                            last_letter_time = current_time
                            print(f"Letter: {most_common} | Word: {current_word}")

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
                speak(sentence)
        elif key == ord('c'):
            current_word = ""
            sentence     = ""
            print("Cleared!")

    cv2.destroyAllWindows()
    print("Goodbye!")


if __name__ == "__main__":
    main()