import cv2
import numpy as np
import config
from worker_base import WorkerBase

class DetectWorker(WorkerBase):
    def load_model(self):
        self.nn = cv2.dnn.readNetFromDarknet(config.DETECT_CFG_PATH, config.DETECT_WEIGHTS_PATH)
        self.nn.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.nn.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.layer_names = self.nn.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.nn.getUnconnectedOutLayers()]


  # This is what is called by the parent class. frame is at frame, obviously.
    def process_frame(self, payload):
        frame = payload["frame"]
        frame_id = payload["frame_id"]
        timestamp = payload["timestamp"]

        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.nn.setInput(blob)
        outputs = self.nn.forward(self.output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = scores[class_id]
                if confidence > 0.5:
                    print("FOUND ONE!!")
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)


#        print(f"[{self.name}] Processing frame {frame_id} (timestamp {timestamp:.2f}) for person detection.")


