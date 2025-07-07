# face_worker.py

from worker_base import WorkerBase

class FaceWorker(WorkerBase):
    def process_frame(self, payload):
        frame_id = payload["frame_id"]
        timestamp = payload["timestamp"]
        print(f"[{self.name}] Processing frame {frame_id} (timestamp {timestamp:.2f}) for face recognition.")

    def load_model(self):
        pass
