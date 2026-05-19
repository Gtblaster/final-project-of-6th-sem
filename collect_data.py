"""
Step 1 — Data Collection  (mirror + dual-hand aware)
=====================================================
Collects wrist-relative, handedness-normalized landmarks.
Left-hand X coords are flipped so the model sees both hands identically.

Output: data/landmarks.csv
"""

import cv2, csv, os, numpy as np, mediapipe as mp, urllib.request
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

CLASSES      = ["Hello", "ILoveYou", "No", "Please", "Thanks", "Yes"]
SAMPLES_PER  = 300          # frames per class (collect with BOTH hands for free augmentation)
OUTPUT_CSV   = "data/landmarks.csv"
MODEL_PATH   = "hand_landmarker.task"

os.makedirs("data", exist_ok=True)

# ── MediaPipe (up to 2 hands) ──────────────────────────────────────────────────
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

def draw_hand(frame, lms, h, w, color=(0,200,255)):
    pts = [(int(lm.x*w), int(lm.y*h)) for lm in lms]
    for a,b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (0,255,0), -1)

# ── Core preprocessing ─────────────────────────────────────────────────────────
def normalize_landmarks(lms, handedness: str) -> np.ndarray:
    """
    1. Translate all points relative to wrist (landmark 0).
    2. Mirror X for Left hand  →  model always sees a right-hand coordinate system.
    3. Scale by max absolute value for scale/distance invariance.
    Returns flat float32 array of shape (63,).
    """
    wrist  = np.array([lms[0].x, lms[0].y, lms[0].z], dtype=np.float32)
    coords = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
    coords -= wrist                          # wrist-relative

    if handedness.lower() == "left":
        coords[:, 0] *= -1.0                 # mirror X  →  right-hand system

    flat  = coords.flatten()
    scale = np.max(np.abs(flat)) + 1e-6
    return flat / scale                      # scale invariant

def hand_area(lms) -> float:
    """Normalized bounding-box area (0-1). Filters partial/distant hands."""
    xs = [lm.x for lm in lms]; ys = [lm.y for lm in lms]
    return (max(xs)-min(xs)) * (max(ys)-min(ys))

# ── CSV header ─────────────────────────────────────────────────────────────────
with open(OUTPUT_CSV, 'w', newline='') as f:
    header = [f"{ax}{i}" for i in range(21) for ax in ['x','y','z']] + ['label']
    csv.writer(f).writerow(header)

cap      = cv2.VideoCapture(0)
frame_ts = 0

for cls in CLASSES:
    collected = 0
    recording = False
    print(f"\n>>> Sign: {cls}  |  Use EITHER hand  |  SPACE = start, Q = skip")

    while collected < SAMPLES_PER:
        ret, frame = cap.read()
        if not ret: break
        h, w = frame.shape[:2]
        frame_ts += 1

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                          data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = landmarker.detect_for_video(mp_img, frame_ts)

        saved_this_frame = False

        if result.hand_landmarks:
            for idx, (lms, handed) in enumerate(
                    zip(result.hand_landmarks, result.handedness)):

                side  = handed[0].category_name   # "Left" or "Right"
                area  = hand_area(lms)
                color = (0,200,255) if side=="Right" else (255,150,0)
                draw_hand(frame, lms, h, w, color)

                # Spatial filter
                if area < 0.015:
                    continue

                if recording and not saved_this_frame:
                    row = normalize_landmarks(lms, side).tolist() + [cls]
                    with open(OUTPUT_CSV, 'a', newline='') as f:
                        csv.writer(f).writerow(row)
                    collected += 1
                    saved_this_frame = True   # one sample per frame

                # Label each hand
                lx = int(min(lm.x for lm in lms)*w)
                ly = int(min(lm.y for lm in lms)*h) - 10
                cv2.putText(frame, side, (lx, max(15,ly)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # HUD
        status = f"Recording {collected}/{SAMPLES_PER}" if recording else "Press SPACE"
        color  = (0,200,0) if recording else (0,100,255)
        cv2.rectangle(frame, (0,0), (w,50), (0,0,0), -1)
        cv2.putText(frame, f"{cls}  |  {status}", (10,35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow("Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '): recording = True
        if key == ord('q'): break

    print(f"  Saved {collected} samples for {cls}")

cap.release()
cv2.destroyAllWindows()
landmarker.close()
print(f"\nDone! Saved to {OUTPUT_CSV}")
