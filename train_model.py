"""
Step 2 — Train GRU Classifier  (mirror-aware)
==============================================
Reads landmark CSV (already handedness-normalized by collect_data.py).
Builds 15-frame sliding windows → stacked GRU → softmax.

Output: data/sign_model.keras   data/label_classes.npy
"""

import numpy as np, pandas as pd, os, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ── Hyperparameters ────────────────────────────────────────────────────────────
WINDOW_SIZE = 15        # frames per sequence
FEATURE_DIM = 63        # 21 × 3
NUM_CLASSES = 6
BATCH_SIZE  = 32
EPOCHS      = 100
LR          = 1e-3
DROPOUT     = 0.4
GRU_UNITS   = [128, 64]

CSV_PATH    = "data/landmarks.csv"
MODEL_OUT   = "data/sign_model.keras"
ENCODER_OUT = "data/label_classes.npy"

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(CSV_PATH)
print(f"  Frames : {len(df)}")
print(f"  Classes:\n{df['label'].value_counts()}\n")

le = LabelEncoder()
df['enc'] = le.fit_transform(df['label'])
np.save(ENCODER_OUT, le.classes_)

feat_cols = [c for c in df.columns if c not in ('label','enc')]
X = df[feat_cols].values.astype(np.float32)
y = df['enc'].values

# ── Sliding windows ────────────────────────────────────────────────────────────
def make_windows(X, y, win):
    Xw, yw = [], []
    for i in range(len(X) - win + 1):
        Xw.append(X[i:i+win])
        yw.append(y[i+win-1])
    return np.array(Xw, dtype=np.float32), np.array(yw)

Xw, yw = make_windows(X, y, WINDOW_SIZE)
print(f"  Windows: {Xw.shape}")

X_tr, X_val, y_tr, y_val = train_test_split(
    Xw, yw, test_size=0.2, random_state=42, stratify=yw)

cw = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
class_weights = dict(enumerate(cw))
print(f"  Class weights: { {le.classes_[k]:round(v,2) for k,v in class_weights.items()} }\n")

# ── Augmentation dataset ───────────────────────────────────────────────────────
class AugDataset(tf.keras.utils.Sequence):
    def __init__(self, X, y, bs, aug=False):
        self.X, self.y, self.bs, self.aug = X, y, bs, aug
        self.idx = np.arange(len(X))

    def __len__(self): return int(np.ceil(len(self.X)/self.bs))

    def __getitem__(self, i):
        b  = self.idx[i*self.bs:(i+1)*self.bs]
        Xb = self.X[b].copy()
        yb = self.y[b]
        if self.aug:
            # Gaussian noise
            Xb += np.random.normal(0, 0.012, Xb.shape).astype(np.float32)
            # Scale jitter ±10 %
            Xb *= np.random.uniform(0.90, 1.10, (len(Xb),1,1)).astype(np.float32)
            # Random time-shift (roll 0-2 frames)
            shift = np.random.randint(0, 3)
            if shift: Xb = np.roll(Xb, shift, axis=1)
        return Xb, tf.keras.utils.to_categorical(yb, NUM_CLASSES)

    def on_epoch_end(self): np.random.shuffle(self.idx)

tr_ds  = AugDataset(X_tr,  y_tr,  BATCH_SIZE, aug=True)
val_ds = AugDataset(X_val, y_val, BATCH_SIZE, aug=False)

# ── Model ──────────────────────────────────────────────────────────────────────
inp = keras.Input(shape=(WINDOW_SIZE, FEATURE_DIM), name="landmarks")

x = layers.GRU(GRU_UNITS[0], return_sequences=True,
               kernel_regularizer=keras.regularizers.l2(1e-4))(inp)
x = layers.BatchNormalization()(x)
x = layers.Dropout(DROPOUT)(x)

x = layers.GRU(GRU_UNITS[1], return_sequences=False,
               kernel_regularizer=keras.regularizers.l2(1e-4))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(DROPOUT)(x)

x   = layers.Dense(64, activation='relu',
                   kernel_regularizer=keras.regularizers.l2(1e-4))(x)
x   = layers.Dropout(0.3)(x)
out = layers.Dense(NUM_CLASSES, activation='softmax', name="prediction")(x)

model = keras.Model(inp, out)
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=7, min_lr=1e-5, verbose=1),
    keras.callbacks.ModelCheckpoint(
        MODEL_OUT, monitor='val_accuracy', save_best_only=True, verbose=1),
]

print("Training...")
hist = model.fit(tr_ds, validation_data=val_ds,
                 epochs=EPOCHS, callbacks=callbacks,
                 class_weight=class_weights)

print(f"\nBest val accuracy : {max(hist.history['val_accuracy']):.4f}")
print(f"Model  → {MODEL_OUT}")
print(f"Labels → {ENCODER_OUT}")
