from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import cv2
import numpy as np
import joblib
import mediapipe as mp
from collections import deque
import time
import threading
import urllib.request
import pyttsx3

app = Flask(__name__)
CORS(app)

model      = joblib.load('models/best_model.pkl')
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands      = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

state = {
    'predicted_letter': '',
    'current_word':     '',
    'sentence':         '',
    'confidence':       0,
    'hand_detected':    False,
    'camera_mode':      'laptop',
    'motion_mode':      '',
}

prediction_buffer = deque(maxlen=10)
last_letter_time  = time.time()
LETTER_DELAY      = 2.0
CONFIDENCE_FRAMES = 7
PHONE_URL         = "http://192.168.1.3:8080/shot.jpg"
cap               = None
cap_lock          = threading.Lock()
speak_lock        = threading.Lock()

pinky_hold_start  = None
motion_trajectory = []
motion_mode       = None
HOLD_TIME         = 2.0
MAX_TRAJECTORY    = 60


def load_dictionary(path='data/words.txt'):
    try:
        with open(path, 'r') as f:
            words = [w.strip().upper() for w in f.readlines()]
        words = [w for w in words if 2 <= len(w) <= 10]
        return sorted(words)
    except FileNotFoundError:
        print(f"Warning: Dictionary file '{path}' not found. Using fallback dictionary.")
        return ["HELLO", "WORLD", "SIGN", "LANGUAGE", "THANK", "YOU", "PLEASE", "SORRY", "YES", "NO", "HELP", "AGENT"]

DICTIONARY = load_dictionary()
print(f"Dictionary loaded: {len(DICTIONARY)} words")


def get_suggestions(prefix, n=4):
    if not prefix:
        return []
    prefix  = prefix.upper()
    matches = [w for w in DICTIONARY if w.startswith(prefix)]
    return matches[:n]


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


def normalize_landmarks(hand_landmarks):
    row = []
    for landmark in hand_landmarks.landmark:
        row.append(landmark.x)
        row.append(landmark.y)
    wrist_x    = row[0]
    wrist_y    = row[1]
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


def is_finger_up(landmarks, tip_id, base_id):
    return landmarks[tip_id].y < landmarks[base_id].y


def is_pinky_only_up(hand_landmarks):
    lm = hand_landmarks.landmark
    return (
        is_finger_up(lm, 20, 17)     and
        not is_finger_up(lm, 16, 13) and
        not is_finger_up(lm, 12, 9)  and
        not is_finger_up(lm, 8,  5)
    )


def is_pointer_only_up(hand_landmarks):
    lm = hand_landmarks.landmark
    return (
        is_finger_up(lm, 8,  5)      and
        not is_finger_up(lm, 12, 9)  and
        not is_finger_up(lm, 16, 13) and
        not is_finger_up(lm, 20, 17)
    )


def is_hand_still(trajectory, threshold=0.04):
    if len(trajectory) < 10:
        return True
    xs = [p[0] for p in trajectory]
    ys = [p[1] for p in trajectory]
    return (max(xs) - min(xs)) < threshold and \
        (max(ys) - min(ys)) < threshold


def is_J_shape(trajectory, min_distance=0.1):
    if len(trajectory) < 20:
        return False
    xs  = [p[0] for p in trajectory]
    ys  = [p[1] for p in trajectory]
    dy  = ys[-1] - ys[0]
    dx  = xs[-1] - xs[0]
    mid = len(xs) // 2
    second_half_dx = xs[-1] - xs[mid]
    moved_down  = dy > min_distance
    curved_left = dx < 0 or second_half_dx < -0.02
    significant = (abs(dy) + abs(dx)) > min_distance
    return moved_down and curved_left and significant


def is_Z_shape(trajectory, min_distance=0.08):
    if len(trajectory) < 20:
        return False
    n   = len(trajectory)
    s1  = trajectory[:n//3]
    s2  = trajectory[n//3:2*n//3]
    s3  = trajectory[2*n//3:]
    dx1 = s1[-1][0] - s1[0][0]
    dx2 = s2[-1][0] - s2[0][0]
    dx3 = s3[-1][0] - s3[0][0]
    all_xs      = [p[0] for p in trajectory]
    significant = (max(all_xs) - min(all_xs)) > min_distance
    return (dx1 > 0.02 and dx2 < -0.02 and
            dx3 > 0.02 and significant)


def check_J_Z(hand_landmarks, current_time):
    global pinky_hold_start, motion_trajectory, motion_mode

    lm        = hand_landmarks.landmark
    pinky_tip = (lm[20].x, lm[20].y)
    index_tip = (lm[8].x,  lm[8].y)

    if is_pinky_only_up(hand_landmarks):
        state['motion_mode'] = 'J mode'

        if motion_mode == 'J':
            motion_trajectory.append(pinky_tip)
            if len(motion_trajectory) > MAX_TRAJECTORY:
                motion_trajectory.pop(0)
            if is_J_shape(motion_trajectory):
                motion_trajectory = []
                motion_mode       = None
                pinky_hold_start  = None
                state['motion_mode'] = ''
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
                    pinky_hold_start     = None
                    motion_trajectory    = []
                    motion_mode          = None
                    state['motion_mode'] = ''
                    return 'I'

    elif is_pointer_only_up(hand_landmarks):
        state['motion_mode'] = 'Z mode'

        if motion_mode == 'Z':
            motion_trajectory.append(index_tip)
            if len(motion_trajectory) > MAX_TRAJECTORY:
                motion_trajectory.pop(0)
            if is_Z_shape(motion_trajectory):
                motion_trajectory    = []
                motion_mode          = None
                state['motion_mode'] = ''
                return 'Z'

        elif not is_hand_still(motion_trajectory):
            motion_mode = 'Z'
            motion_trajectory.append(index_tip)

        else:
            motion_trajectory    = [index_tip]
            motion_mode          = None

    else:
        pinky_hold_start     = None
        motion_trajectory    = []
        motion_mode          = None
        state['motion_mode'] = ''

    return None


def get_phone_frame():
    try:
        img_resp = urllib.request.urlopen(PHONE_URL, timeout=2)
        imgnp    = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        return cv2.imdecode(imgnp, -1)
    except:
        return None


def get_laptop_frame():
    global cap
    with cap_lock:
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        return frame if ret else None


def process_frame(frame):
    global last_letter_time

    frame     = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results   = hands.process(image_rgb)

    state['hand_detected'] = results.multi_hand_landmarks is not None

    if state['hand_detected']:
        for hand_lms in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0),   thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)
            )

        hand_lm      = results.multi_hand_landmarks[0]
        current_time = time.time()

        special = check_J_Z(hand_lm, current_time)

        if special:
            state['predicted_letter'] = special
            state['confidence']       = 100
            prediction_buffer.clear()
            if current_time - last_letter_time > LETTER_DELAY:
                state['current_word'] += special
                last_letter_time       = current_time
        else:
            landmarks  = normalize_landmarks(hand_lm)
            prediction = model.predict(landmarks)[0]
            prediction_buffer.append(prediction)

            if len(prediction_buffer) == 10:
                most_common = max(set(prediction_buffer),
                                key=list(prediction_buffer).count)
                count       = list(prediction_buffer).count(most_common)
                state['confidence'] = round((count / 10) * 100)

                if count >= CONFIDENCE_FRAMES:
                    state['predicted_letter'] = most_common
                    if current_time - last_letter_time > LETTER_DELAY:
                        state['current_word'] += most_common
                        last_letter_time       = current_time
    else:
        prediction_buffer.clear()
        state['predicted_letter'] = ''
        state['confidence']       = 0
        state['motion_mode']      = ''

    return frame


def generate_frames():
    while True:
        frame = get_phone_frame() if state['camera_mode'] == 'phone' else get_laptop_frame()
        if frame is None:
            continue
        frame       = cv2.resize(frame, (640, 480))
        frame       = process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            buffer.tobytes() + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/state')
def get_state():
    return jsonify(state)

@app.route('/add_word', methods=['POST'])
def add_word():
    if state['current_word']:
        word                  = state['current_word']
        state['sentence']    += word + ' '
        state['current_word'] = ''
        prediction_buffer.clear()
        speak(word)
    return jsonify({'success': True})

@app.route('/speak', methods=['POST'])
def speak_sentence():
    sentence = state['sentence'].strip()
    if sentence:
        speak(sentence)
    return jsonify({'success': True})

@app.route('/clear', methods=['POST'])
def clear():
    state['current_word'] = ''
    state['sentence']     = ''
    prediction_buffer.clear()
    return jsonify({'success': True})

@app.route('/delete_letter', methods=['POST'])
def delete_letter():
    if state['current_word']:
        state['current_word'] = state['current_word'][:-1]
    return jsonify({'success': True})

@app.route('/switch_camera', methods=['POST'])
def switch_camera():
    global cap
    state['camera_mode'] = request.get_json().get('mode', 'laptop')
    if state['camera_mode'] == 'laptop':
        with cap_lock:
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(0)
    else:
        with cap_lock:
            if cap:
                cap.release()
                cap = None
    prediction_buffer.clear()
    return jsonify({'success': True})

@app.route('/suggestions')
def suggestions():
    prefix  = request.args.get('prefix', '')
    results = get_suggestions(prefix)
    return jsonify({'suggestions': results})

@app.route('/use_suggestion', methods=['POST'])
def use_suggestion():
    word = request.get_json().get('word', '')
    if word:
        state['sentence']    += word + ' '
        state['current_word'] = ''
        prediction_buffer.clear()
        speak(word)
    return jsonify({'success': True})


if __name__ == '__main__':
    print("Open browser at: http://localhost:5000")
    app.run(debug=False, threaded=True)