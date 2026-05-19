"""
Step 3 — Real-Time Inference  (mirror + dual-hand)
===================================================
• Left-hand X coords are mirrored → same model for both hands
• Each hand runs inference independently through its own frame buffer
• Dual-hand output shown simultaneously
• Single-output mode: prioritizes hand closest to frame center
  (or higher tracking confidence if tied)
"""

import cv2, numpy as np, mediapipe as mp, os, urllib.request, tensorflow as tf
from collections import deque
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_PATH    = "hand_landmarker.task"
SIGN_MODEL    = "data/sign_model.keras"
LABEL_CLASSES = "data/label_classes.npy"

WINDOW_SIZE   = 15      # must match training
MIN_CONF      = 0.70    # softmax threshold to accept a prediction
MIN_HAND_AREA = 0.015   # normalized bbox area filter
HOLD_FRAMES   = 8       # frames new winner must dominate before switching label

# ── Load sign model ────────────────────────────────────────────────────────────
if not os.path.exists(SIGN_MODEL):
    print("ERROR: model not found. Run collect_data.py then train_model.py first.")
    exit(1)

print("Loading sign model...")
sign_model = tf.keras.models.load_model(SIGN_MODEL)
classes    = np.load(LABEL_CLASSES, allow_pickle=True)
print(f"  Classes: {classes}")

# ── MediaPipe (2 hands) ────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task", MODEL_PATH)

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
hand_options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.70,
    min_hand_presence_confidence=0.70,
    min_tracking_confidence=0.70,
    running_mode=mp_vision.RunningMode.VIDEO
)
landmarker = mp_vision.HandLandmarker.create_from_options(hand_options)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17)
]

# ── Preprocessing ──────────────────────────────────────────────────────────────
def normalize_landmarks(lms, handedness: str) -> np.ndarray:
    """
    Wrist-relative normalization with left-hand X mirroring.
    Each hand is normalized strictly within its own local coordinate system
    (wrist origin) — no cross-hand landmark bleed.

    Steps:
      1. Subtract wrist (landmark 0) from all 21 points.
      2. If Left hand: X_norm = -1 * X_relative  (mirror to right-hand system).
      3. Scale by max absolute value for scale/distance invariance.
    """
    wrist  = np.array([lms[0].x, lms[0].y, lms[0].z], dtype=np.float32)
    coords = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)

    # Step 1 — translate to wrist origin (spatial anchor separation)
    coords -= wrist

    # Step 2 — mirror left hand X so both hands share the same coordinate system
    if handedness.lower() == "left":
        coords[:, 0] *= -1.0

    # Step 3 — scale invariance
    flat  = coords.flatten()
    scale = np.max(np.abs(flat)) + 1e-6
    return (flat / scale).astype(np.float32)

def hand_center(lms, w, h):
    """Returns (cx, cy) in pixels."""
    xs = [lm.x*w for lm in lms]; ys = [lm.y*h for lm in lms]
    return (np.mean(xs), np.mean(ys))

def hand_area(lms):
    xs = [lm.x for lm in lms]; ys = [lm.y for lm in lms]
    return (max(xs)-min(xs)) * (max(ys)-min(ys))

# ── Per-hand state tracker ─────────────────────────────────────────────────────
class HandTracker:
    """
    Independent inference state for one hand.
    Maintains its own frame buffer, candidate label, and hold counter.
    """
    def __init__(self, label: str):
        self.label          = label          # "Left" or "Right"
        self.buffer         = deque(maxlen=WINDOW_SIZE)
        self.candidate      = None
        self.candidate_cnt  = 0
        self.displayed      = None
        self.displayed_conf = 0.0
        self.last_probs     = np.zeros(len(classes))

    def push(self, feat: np.ndarray):
        self.buffer.append(feat)

    def predict(self) -> tuple:
        """Returns (label_str, confidence) or (None, 0)."""
        if len(self.buffer) < WINDOW_SIZE:
            return None, 0.0

        seq   = np.array(self.buffer, dtype=np.float32)[np.newaxis]   # (1,15,63)
        probs = sign_model.predict(seq, verbose=0)[0]
        self.last_probs = probs

        top_i = int(np.argmax(probs))
        top_p = float(probs[top_i])
        pred  = classes[top_i]

        if top_p >= MIN_CONF:
            if pred == self.candidate:
                self.candidate_cnt += 1
            else:
                self.candidate     = pred
                self.candidate_cnt = 1

            if self.candidate_cnt >= HOLD_FRAMES:
                self.displayed      = self.candidate
                self.displayed_conf = top_p

        return self.displayed, self.displayed_conf

    def reset(self):
        self.buffer.clear()
        self.candidate     = None
        self.candidate_cnt = 0
        self.displayed     = None
        self.displayed_conf = 0.0
        self.last_probs    = np.zeros(len(classes))

# Two independent trackers — one per hand slot
trackers = {"Left": HandTracker("Left"), "Right": HandTracker("Right")}

# ── Drawing helpers ────────────────────────────────────────────────────────────
CLASS_COLORS = {
    "Hello":    (0, 200, 50),
    "ILoveYou": (200, 0, 200),
    "No":       (0, 50, 220),
    "Please":   (200, 140, 0),
    "Thanks":   (0, 180, 180),
    "Yes":      (0, 140, 255),
}
HAND_COLORS = {"Left": (255, 150, 0), "Right": (0, 200, 255)}

def draw_hand(frame, lms, h, w, color):
    pts = [(int(lm.x*w), int(lm.y*h)) for lm in lms]
    for a,b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (0,255,0), -1)

def draw_label(frame, text, x1, y1, w, color):
    tw = len(text)*16
    by = max(40, y1)
    cv2.rectangle(frame, (x1, by-38), (min(w, x1+tw), by), color, -1)
    cv2.putText(frame, text, (x1+5, by-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

def draw_bar_chart(frame, probs, offset_y, w, title):
    bx = w - 215
    cv2.rectangle(frame, (bx-5, offset_y), (w-5, offset_y+len(classes)*28+20), (20,20,20), -1)
    cv2.putText(frame, title, (bx, offset_y+14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    for i, cls_name in enumerate(classes):
        bar = int(probs[i]*190)
        bc  = CLASS_COLORS.get(cls_name, (150,150,150))
        y   = offset_y + 20 + i*28
        cv2.rectangle(frame, (bx, y), (bx+bar, y+18), bc, -1)
        cv2.putText(frame, f"{cls_name[:8]} {probs[i]:.0%}",
                    (bx, y+14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,255), 1)

# ── Webcam ─────────────────────────────────────────────────────────────────────
cap      = cv2.VideoCapture(0)
frame_ts = 0
print("Running. Press Q to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    h, w = frame.shape[:2]
    frame_ts += 1

    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                      data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = landmarker.detect_for_video(mp_img, frame_ts)

    active_sides = set()

    # ── Per-hand inference loop ────────────────────────────────────────────────
    hand_results = []   # [(side, lms, label, conf, tracking_score, cx, cy)]

    if result.hand_landmarks:
        for lms, handed in zip(result.hand_landmarks, result.handedness):
            side  = handed[0].category_name          # "Left" or "Right"
            score = handed[0].score                  # tracking confidence
            area  = hand_area(lms)

            # Spatial filter — ignore tiny/partial hands
            if area < MIN_HAND_AREA:
                continue

            active_sides.add(side)
            tracker = trackers[side]

            # Normalize strictly within this hand's own local bounding box
            feat = normalize_landmarks(lms, side)
            tracker.push(feat)
            label, conf = tracker.predict()

            cx, cy = hand_center(lms, w, h)
            hand_results.append((side, lms, label, conf, score, cx, cy))

            # Draw skeleton
            draw_hand(frame, lms, h, w, HAND_COLORS[side])

            # Bounding box
            xs = [lm.x*w for lm in lms]; ys = [lm.y*h for lm in lms]
            x1 = max(0, int(min(xs))-30); y1 = max(0, int(min(ys))-30)
            x2 = min(w, int(max(xs))+30); y2 = min(h, int(max(ys))+30)
            box_color = CLASS_COLORS.get(label, HAND_COLORS[side])
            cv2.rectangle(frame, (x1,y1), (x2,y2), box_color, 2)

            # Hand side tag
            cv2.putText(frame, side, (x1, max(15, y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, HAND_COLORS[side], 2)

            # Prediction label
            if label:
                draw_label(frame, f"{label} {conf:.0%}", x1, y1, w, box_color)

    # ── Reset trackers for hands that disappeared ──────────────────────────────
    for side, tracker in trackers.items():
        if side not in active_sides:
            tracker.reset()

    # ── Dual-hand summary bar ──────────────────────────────────────────────────
    if len(hand_results) == 2:
        r0 = hand_results[0]; r1 = hand_results[1]
        summary = (f"L:{r0[2] or '?'}  R:{r1[2] or '?'}"
                   if r0[0]=="Left"
                   else f"L:{r1[2] or '?'}  R:{r0[2] or '?'}")
        cv2.rectangle(frame, (0, h-45), (w, h), (0,0,0), -1)
        cv2.putText(frame, summary, (10, h-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

    # ── Single-output mode: dynamic hand prioritization ────────────────────────
    # Priority 1: hand closest to frame center
    # Priority 2: higher tracking confidence (tiebreak)
    if hand_results:
        cx_frame, cy_frame = w/2, h/2
        def priority_score(r):
            side, lms, label, conf, track_score, cx, cy = r
            dist = ((cx-cx_frame)**2 + (cy-cy_frame)**2) ** 0.5
            return dist - track_score * 50   # lower = better

        best = min(hand_results, key=priority_score)
        side_b, _, label_b, conf_b, _, _, _ = best
        if label_b:
            tag = f"Primary ({side_b}): {label_b}  {conf_b:.0%}"
            cv2.rectangle(frame, (0,0), (len(tag)*14+10, 42), (0,0,0), -1)
            cv2.putText(frame, tag, (8, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        CLASS_COLORS.get(label_b,(255,255,255)), 2)

    # ── Confidence bar charts (one per active hand) ────────────────────────────
    for i, (side, lms, label, conf, score, cx, cy) in enumerate(hand_results):
        draw_bar_chart(frame, trackers[side].last_probs,
                       offset_y=5 + i*(len(classes)*28+30),
                       w=w, title=f"{side} hand")

    if not hand_results:
        cv2.putText(frame, "Show your hand...", (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

    # Buffer fill indicator
    for i, (side, *_) in enumerate(hand_results):
        buf = len(trackers[side].buffer)
        cv2.putText(frame, f"{side} buf:{buf}/{WINDOW_SIZE}",
                    (10, h-50-i*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)

    cv2.imshow("Sign Language Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
