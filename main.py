import os
import cv2
import numpy as np
import time
import threading
from collections import deque, Counter
import tensorflow as tf
from tensorflow import keras

# ---------------------------
# Suppress TF Warnings
# ---------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ---------------------------
# Model Paths
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "keras_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")

# Check model existence
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")
if not os.path.isfile(LABELS_PATH):
    raise FileNotFoundError(f"❌ Labels not found: {LABELS_PATH}")

# Load model + labels
model = keras.models.load_model(MODEL_PATH)

def parse_label(line):
    parts = line.split()
    return " ".join(parts[1:]) if parts[0].isdigit() else line

with open(LABELS_PATH, "r") as f:
    class_names = [parse_label(line.strip()) for line in f if line.strip()]

print(f"✅ Loaded model with {len(class_names)} classes")

# ---------------------------
# Text-to-Speech
# ---------------------------
def speak(word):
    def _speak():
        safe = word.replace("'", "''")
        command = (
            "powershell -Command "
            "\"Add-Type -AssemblyName System.Speech; "
            "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.Speak('{safe}')\""
        )
        os.system(command)
    threading.Thread(target=_speak, daemon=True).start()


# ---------------------------
# Camera Setup
# ---------------------------
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
camera.set(3, 640)
camera.set(4, 480)

if not camera.isOpened():
    raise RuntimeError("Camera not detected!")

# ROI box
x, y, w, h = 208, 128, 224, 224

# ---------------------------
# Prediction variables
# ---------------------------
frame_buffer = deque(maxlen=8)
predicted_letters = []
sentence = []
last_add_time = 0
interval = 2.0
confidence_threshold = 0.90
last_spoken_word = ""

print("\n📌 Press SPACE to finalize word\n📌 Press C to clear\n📌 Press ESC to quit\n")


# ---------------------------
# Main Loop
# ---------------------------
while True:
    ret, frame = camera.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    display = frame.copy()

    # Draw ROI
    roi = frame[y:y+h, x:x+w]
    cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Model preprocessing
    resized = cv2.resize(roi, (224, 224))
    img = np.asarray(resized, dtype=np.float32).reshape(1, 224, 224, 3)
    img = (img / 127.5) - 1

    preds = model.predict(img, verbose=0)
    index = int(np.argmax(preds))
    label = class_names[index]
    confidence = float(preds[0][index])

    frame_buffer.append((label, confidence))

    hand_detected = False
    chosen_letter = ""

    if len(frame_buffer) == frame_buffer.maxlen:
        top = Counter([p[0] for p in frame_buffer]).most_common(1)[0][0]
        confs = [p[1] for p in frame_buffer if p[0] == top]
        avg_conf = float(np.mean(confs))

        if top.lower() != "nothing" and avg_conf >= confidence_threshold:
            hand_detected = True
            chosen_letter = top

        cv2.putText(display, f"Prediction: {top} ({round(avg_conf*100)}%)",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2)

    if hand_detected and (time.time() - last_add_time > interval):
        predicted_letters.append(chosen_letter)
        last_add_time = time.time()

    # UI Text
    if not hand_detected:
        cv2.putText(display, "No hand detected", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

    word = "".join(predicted_letters)
    cv2.putText(display, f"Word: {word}",
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 2)
    cv2.putText(display, f"Sentence: {' '.join(sentence)}",
                (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 0, 180), 2)

    cv2.imshow("Sign Language to Text", display)

    # Keys
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break

    if key == ord('c'):
        predicted_letters.clear()
        sentence.clear()

    if key == 8:  # Backspace
        if predicted_letters:
            predicted_letters.pop()

    if key == 32:  # SPACE → finalize word
        if word:
            sentence.append(word)
            speak(word)  # ✅ Speak full word once
            predicted_letters.clear()


camera.release()
cv2.destroyAllWindows()
print("\n✅ Program Closed Successfully")
