import cv2
import numpy as np
import config
from worker_base import WorkerBase

class DetectWorker(WorkerBase):
    def __init__(self, input_queue, stop_event, face_output_queue, name="DetectWorker"):
        self.face_output_queue = face_output_queue
        super().__init__(input_queue, stop_event, name)

    def load_model(self):
        self.nn = cv2.dnn.readNetFromDarknet(config.DETECT_CFG_PATH, config.DETECT_WEIGHTS_PATH)
        self.nn.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.nn.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.layer_names = self.nn.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.nn.getUnconnectedOutLayers()]

    def process_frame(self, payload):
        frame = payload["frame"]
        frame_id = payload["frame_id"]
        timestamp = payload["timestamp"]

        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.nn.setInput(blob)
        outputs = self.nn.forward(self.output_layers)

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = max(0, int(center_x - w / 2))
                    y = max(0, int(center_y - h / 2))
                    w = min(w, width - x)
                    h = min(h, height - y)

                    if class_id == 1:  # face class
                        face_crop = frame[y:y+h, x:x+w]
                        if face_crop.size == 0:
                            continue
                        face_resized = cv2.resize(face_crop, (150, 150))

                        try:
                            self.face_output_queue.put_nowait({
                                "frame_id": frame_id,
                                "timestamp": timestamp,
                                "face": face_resized,
                            })
                        except:
                            print(f"[{self.name}] Warning: face queue full, face dropped.")
