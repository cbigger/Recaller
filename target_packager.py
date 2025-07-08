import os
import cv2
import numpy as np
import argparse
import config


class FacePackager:
    def __init__(self, input_dir, output_dir, confidence_threshold=0.5, face_class_id=1, nms_mode=False):
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

def main():
    parser = argparse.ArgumentParser(description="Extract face crops from images using YOLO-based detection.")
    parser.add_argument("--build_test", action="store_true", help="Export to off-target face dataset instead of target set.")
    parser.add_argument("--no_nms", action="store_true", help="Export to off-target face dataset instead of target set.")
    args = parser.parse_args()

    input_dir = config.INPUT_DIR
    output_dir = config.OUTPUT_DIR

    if args.build_test:
        input_dir = config.OFFTARGET_INPUT_DIR
        output_dir = config.OFFTARGET_OUTPUT_DIR
        packager = FacePackager(input_dir, output_dir, nms_mode=True)
    elif args.no_nms:
        packager = FacePackager(input_dir, output_dir, nms_mode=False)
    else:
        packager = FacePackager(input_dir, output_dir, nms_mode=True)

    packager.run()


if __name__ == "__main__":
    main()
