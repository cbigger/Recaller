import cv2
import os
import numpy as np
#import config as cfg

# --- Configuration ---
TRAIN_DIR = "target"
TEST_DIR = "offtarget"
FACE_SIZE = (150, 150)
MODEL_PATH = "target/face_model.xml"
LABEL_ID = 1

# --- Load Training Images ---
def load_images_from_folder(folder, label_id):
    images = []
    labels = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img_resized = cv2.resize(img, FACE_SIZE)
        images.append(img_resized)
        labels.append(label_id)
    return images, labels

print("[INFO] Loading training data...")
train_images, train_labels = load_images_from_folder(TRAIN_DIR, LABEL_ID)

# --- Train Recognizer ---
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(train_images, np.array(train_labels))
recognizer.save(MODEL_PATH)
print(f"[INFO] Model trained and saved to {MODEL_PATH}")

# --- Load and Test Against New Images ---
print("[INFO] Testing recognition on test images...")
for filename in os.listdir(TEST_DIR):
    path = os.path.join(TEST_DIR, filename)
    test_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if test_img is None:
        print(f"[WARN] Could not read {filename}")
        continue

    test_img_resized = cv2.resize(test_img, FACE_SIZE)
    label, confidence = recognizer.predict(test_img_resized)

    print(f"[TEST] {filename}: predicted label={label}, confidence={confidence:.2f}")
