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

engine = pyttsx3.init()
engine.setProperty('rate', 150)

state = {
    'predicted_letter': '',
    'current_word':     '',
    'sentence':         '',
    'confidence':       0,
    'hand_detected':    False,
    'camera_mode':      'laptop',
}

prediction_buffer = deque(maxlen=10)
last_letter_time  = time.time()
LETTER_DELAY      = 2.0
CONFIDENCE_FRAMES = 7
PHONE_URL         = "http://192.168.1.3:8080/shot.jpg"
cap               = None
cap_lock          = threading.Lock()


def speak(text):
    def run():
        engine.say(text)
        engine.runAndWait()
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
        landmarks  = normalize_landmarks(results.multi_hand_landmarks[0])
        prediction = model.predict(landmarks)[0]
        prediction_buffer.append(prediction)

        if len(prediction_buffer) == 10:
            most_common = max(set(prediction_buffer),
                              key=list(prediction_buffer).count)
            count       = list(prediction_buffer).count(most_common)
            state['confidence'] = round((count / 10) * 100)

            if count >= CONFIDENCE_FRAMES:
                state['predicted_letter'] = most_common
                current_time = time.time()
                if current_time - last_letter_time > LETTER_DELAY:
                    state['current_word'] += most_common
                    last_letter_time       = current_time
    else:
        prediction_buffer.clear()
        state['predicted_letter'] = ''
        state['confidence']       = 0

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
        state['sentence']    += state['current_word'] + ' '
        speak(state['current_word'])
        state['current_word'] = ''
    return jsonify({'success': True})

@app.route('/speak', methods=['POST'])
def speak_sentence():
    if state['sentence']:
        speak(state['sentence'])
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


if __name__ == '__main__':
    print("Open browser at: http://localhost:5000")
    app.run(debug=False, threaded=True)