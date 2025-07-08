import os
import cv2
import numpy as np
import argparse
import config

### TARGETTER MODE CLASSES
class FacePackager:
    """
    This class is used with targetting mode to extract faces from still images and store them for training or testing.
    It accepts an argument to turn off NMS which will allow overlapping detections (as in, it will pull multiple crops, usually max of two,
    from a single image/face.)
    """
    def __init__(self, input_dir, output_dir, confidence_threshold=0.5, face_class_id=1, nms_mode=True):

        self.input_dir = input_dir
        self.output_dir = output_dir
        self.confidence_threshold = confidence_threshold
        self.face_class_id = face_class_id
        self.image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        self.nms_mode = nms_mode
        os.makedirs(self.output_dir, exist_ok=True)
        self._load_model()

    def _load_model(self):
        self.net = cv2.dnn.readNetFromDarknet(config.DETECT_CFG_PATH, config.DETECT_WEIGHTS_PATH)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def run(self):
        count = 0

        for filename in os.listdir(self.input_dir):
            if not any(filename.lower().endswith(ext) for ext in self.image_extensions):
                continue

            path = os.path.join(self.input_dir, filename)
            image = cv2.imread(path)
            if image is None:
                print(f"[WARN] Failed to read {filename}")
                continue

            height, width = image.shape[:2]
            blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            self.net.setInput(blob)
            outputs = self.net.forward(self.output_layers)

            boxes = []
            confidences = []

            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = int(np.argmax(scores))
                    confidence = scores[class_id]

                    if confidence < self.confidence_threshold or class_id != self.face_class_id:
                        continue

                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = max(0, int(center_x - w / 2))
                    y = max(0, int(center_y - h / 2))
                    w = min(w, width - x)
                    h = min(h, height - y)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))

            if not self.nms_mode:
                for box in boxes:
                    x, y, w, h = box
                    face_crop = image[y:y + h, x:x + w]
                    if face_crop.size == 0:
                        continue
                    face_resized = cv2.resize(face_crop, (150, 150))
                    out_name = f"face_{count:04d}.jpg"
                    cv2.imwrite(os.path.join(self.output_dir, out_name), face_resized)
                    count += 1
            else:
                indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, 0.3)
                for i in indices:
                    i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
                    x, y, w, h = boxes[i]
                    face_crop = image[y:y + h, x:x + w]
                    if face_crop.size == 0:
                        continue
                    face_resized = cv2.resize(face_crop, (150, 150))
                    out_name = f"face_{count:04d}.jpg"
                    cv2.imwrite(os.path.join(self.output_dir, out_name), face_resized)
                    count += 1

            print(f"[DONE] Saved {count} face images to {self.output_dir}")


class FaceTrainer:
    """
    This class handles the building and testing of target packages. It probably needs to be merged into a unified class with the above.
    Parameters:
    train_dir == target, the directory holding the processed bounded faces of the target
    test_dir  == offtarget, the directory holding the processed bounded faces to test against
    face_model_output_path == target as well.
    """
    def __init__(self, train_dir, test_dir, face_model_output_path, face_size=(150, 150), label_id=1):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.face_model_output_path = face_model_output_path #+ "/target_face.xml"
        self.face_size = face_size
        self.label_id = label_id
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()

    def load_images_from_folder(self, folder, label_id):
        images = []
        labels = []
        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img_resized = cv2.resize(img, self.face_size)
            images.append(img_resized)
            labels.append(label_id)
        return images, labels

    def train(self):
        print("[INFO] Loading training data...")
        train_images, train_labels = self.load_images_from_folder(self.train_dir, self.label_id)
        self.recognizer.train(train_images, np.array(train_labels))
        self.recognizer.save(self.face_model_output_path)
        print(f"[INFO] Model trained and saved to {self.face_model_output_path}")

    def test(self):
        print("[INFO] Testing recognition on test images...")
        for filename in os.listdir(self.test_dir):
            path = os.path.join(self.test_dir, filename)
            test_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if test_img is None:
                print(f"[WARN] Could not read {filename}")
                continue
            test_img_resized = cv2.resize(test_img, self.face_size)
            label, confidence = self.recognizer.predict(test_img_resized)
            print(f"[TEST] {filename}: predicted label={label}, confidence={confidence:.2f}")
