import threading
import time
from queue import Queue, Full
from camera_stream import CameraStream

class FrameProducer(threading.Thread):
    def __init__(self, camera: CameraStream, output_queues: list, stop_event: threading.Event, drop_old=True):
        super().__init__(daemon=True)
        self.camera = camera
        self.queues = output_queues
        self.stop_event = stop_event
        self.drop_old = drop_old
        self.frame_id = 0

    def run(self):
        print("[Producer] Starting frame capture.")
        while not self.stop_event.is_set():
            try:
                frame = self.camera.read_frame()
            except RuntimeError as e:
                print(f"[Producer] Error reading frame: {e}")
                break

            timestamp = time.time()
            payload = {
                "frame": frame,
                "timestamp": timestamp,
                "frame_id": self.frame_id,
            }
            self.frame_id += 1

            for q in self.queues:
                try:
                    if self.drop_old and q.full():
                        try:
                            q.get_nowait()
                        except:
                            pass
                    q.put_nowait(payload)
                except Full:
                    print("[Producer] Warning: Queue full, frame dropped.")

        print("[Producer] Stopped.")
