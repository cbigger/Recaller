import threading
from queue import Queue, Empty

class WorkerBase(threading.Thread):
    def __init__(self, input_queue: Queue, stop_event: threading.Event, name="Worker"):
        super().__init__(daemon=True, name=name)
        self.queue = input_queue
        self.stop_event = stop_event
        self.name = name
        self.load_model()

    def run(self):
        print(f"[{self.name}] Starting.")
        while not self.stop_event.is_set():
            try:
                payload = self.queue.get(timeout=0.1)
                self.process_frame(payload)
            except Empty:
                continue
            except Exception as e:
                print(f"[{self.name}] Error: {e}")

        print(f"[{self.name}] Stopped.")

    def process_frame(self, payload):
        raise NotImplementedError("Subclasses must implement process_frame().")
