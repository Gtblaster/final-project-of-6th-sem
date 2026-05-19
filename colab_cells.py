# ============================================================
# SIGN LANGUAGE RECOGNITION - GOOGLE COLAB
# Copy each CELL block into a separate Colab code cell
# ============================================================

# ── CELL 1: Install ──────────────────────────────────────────
# !pip install mediapipe tensorflow scikit-learn opencv-python-headless -q

# ── CELL 2: Clone repo ───────────────────────────────────────
# !git clone https://github.com/Gtblaster/final-project-of-6th-sem.git
# %cd final-project-of-6th-sem
# !mkdir -p data

# ── CELL 3: Setup camera + MediaPipe ────────────────────────
# import os, csv, urllib.request, time
# import numpy as np
# import cv2
# import mediapipe as mp
# from mediapipe.tasks import python as mp_python
# from mediapipe.tasks.python import vision as mp_vision
# from google.colab.output import eval_js
# from base64 import b64decode
#
# MODEL_PATH = 'hand_landmarker.task'
# if not os.path.exists(MODEL_PATH):
#     urllib.request.urlretrieve(
#         'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
#         MODEL_PATH)
#
# base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
# hand_options = mp_vision.HandLandmarkerOptions(
#     base_options=base_options, num_hands=1,
#     min_hand_detection_confidence=0.70,
#     min_hand_presence_confidence=0.70,
#     min_tracking_confidence=0.70,
#     running_mode=mp_vision.RunningMode.IMAGE)
# landmarker = mp_vision.HandLandmarker.create_from_options(hand_options)
#
# JS = ('async function snap(){'
#       'const s=await navigator.mediaDevices.getUserMedia({video:true});'
#       'const v=document.createElement("video");'
#       'v.srcObject=s;'
#       'await new Promise(r=>{v.onloadedmetadata=r;});'
#       'v.play();'
#       'await new Promise(r=>setTimeout(r,500));'
#       'const c=document.createElement("canvas");'
#       'c.width=v.videoWidth;c.height=v.videoHeight;'
#       'c.getContext("2d").drawImage(v,0,0);'
#       's.getTracks().forEach(t=>t.stop());'
#       'return c.toDataURL("image/jpeg",0.8);}snap();')
#
# def capture_frame():
#     url = eval_js(JS)
#     arr = np.frombuffer(b64decode(url.split(",")[1]), dtype=np.uint8)
#     return cv2.imdecode(arr, cv2.IMREAD_COLOR)
#
# def normalize(lms, side):
#     w = np.array([lms[0].x, lms[0].y, lms[0].z], dtype=np.float32)
#     c = np.array([[l.x,l.y,l.z] for l in lms], dtype=np.float32) - w
#     if side.lower() == 'left':
#         c[:,0] *= -1
#     f = c.flatten()
#     return f / (np.max(np.abs(f)) + 1e-6)
#
# def get_landmarks(frame):
#     img = mp.Image(image_format=mp.ImageFormat.SRGB,
#                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     r = landmarker.detect(img)
#     if not r.hand_landmarks:
#         return None, None
#     lms  = r.hand_landmarks[0]
#     side = r.handedness[0][0].category_name
#     xs = [l.x for l in lms]; ys = [l.y for l in lms]
#     if (max(xs)-min(xs))*(max(ys)-min(ys)) < 0.015:
#         return None, None
#     return normalize(lms, side), side
#
# print('Setup done')

# ── CELL 4: Collect data ─────────────────────────────────────
# from IPython.display import clear_output
# CLASSES = ['Hello','ILoveYou','No','Please','Thanks','Yes']
# SAMPLES = 150
# CSV     = 'data/landmarks.csv'
# with open(CSV,'w',newline='') as f:
#     csv.writer(f).writerow([a+str(i) for i in range(21) for a in ['x','y','z']]+['label'])
# for cls in CLASSES:
#     print('Sign: '+cls+' - hold sign, starting in 3s')
#     time.sleep(3)
#     n=0
#     while n < SAMPLES:
#         frame = capture_frame()
#         feat, side = get_landmarks(frame)
#         if feat is not None:
#             with open(CSV,'a',newline='') as f:
#                 csv.writer(f).writerow(feat.tolist()+[cls])
#             n+=1
#         clear_output(wait=True)
#         print(cls+': '+str(n)+'/'+str(SAMPLES))
#     print(cls+' done')
# print('Collection complete')

# ── CELL 5: Train ────────────────────────────────────────────
# !python train_model.py

# ── CELL 6: Live inference ───────────────────────────────────
# import tensorflow as tf, matplotlib.pyplot as plt
# from collections import deque
# from IPython.display import clear_output
# WINDOW=15; MIN_CONF=0.70; HOLD=8; FRAMES=300
# model   = tf.keras.models.load_model('data/sign_model.keras')
# classes = np.load('data/label_classes.npy', allow_pickle=True)
# COLORS  = {'Hello':'#00C832','ILoveYou':'#C800C8','No':'#DC0032',
#            'Please':'#C88C00','Thanks':'#00B4B4','Yes':'#008CFF'}
# buf=deque(maxlen=WINDOW); cand=None; cnt=0; disp=None; conf=0.0
# for i in range(FRAMES):
#     frame=capture_frame(); feat,side=get_landmarks(frame)
#     if feat is not None:
#         buf.append(feat)
#         if len(buf)==WINDOW:
#             p=model.predict(np.array(buf)[np.newaxis],verbose=0)[0]
#             ti=int(np.argmax(p)); tp=float(p[ti]); pred=classes[ti]
#             if tp>=MIN_CONF:
#                 cand,cnt=(pred,cnt+1) if pred==cand else (pred,1)
#                 if cnt>=HOLD: disp,conf=pred,tp
#             clear_output(wait=True)
#             fig,(a1,a2)=plt.subplots(1,2,figsize=(12,5))
#             a1.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)); a1.axis('off')
#             a1.set_title((disp+' '+str(round(conf*100))+'%') if disp else 'Detecting...',
#                          fontsize=22,fontweight='bold',
#                          color=COLORS.get(disp,'white') if disp else 'red')
#             a2.barh(list(classes),list(p),color=[COLORS.get(c,'#888') for c in classes])
#             a2.set_xlim(0,1); a2.set_title('Confidence')
#             plt.tight_layout(); plt.show()
#     else:
#         buf.clear(); cand=None; cnt=0; disp=None
#         clear_output(wait=True); print('No hand - frame '+str(i))
# print('Done')
