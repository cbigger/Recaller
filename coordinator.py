import threading
from queue import Queue
from camera_stream import CameraStream
from producer import FrameProducer
from face_worker import FaceWorker
from detect_worker import DetectWorker
import config

def run_system():
    stop_event = threading.Event()
    camera = CameraStream()

    detect_queue = Queue(maxsize=config.DETECT_QUEUE_SIZE)
    face_queue = Queue(maxsize=config.FACE_QUEUE_SIZE)

    producer = FrameProducer(camera, [detect_queue], stop_event)
    detect_worker = DetectWorker(detect_queue, stop_event, face_output_queue=face_queue, name="DetectWorker")
    face_worker = FaceWorker(face_queue, stop_event, name="FaceWorker")

    threads = [producer, detect_worker, face_worker]
    for t in threads:
        t.start()

    try:
        print("[Coordinator] System running. Press Ctrl+C to stop.")
        while not stop_event.is_set():
            for t in threads:
                if not t.is_alive():
                    raise RuntimeError(f"{t.name} thread died unexpectedly.")
            threading.Event().wait(1)
    except KeyboardInterrupt:
        print("\n[Coordinator] Shutdown signal received.")
        stop_event.set()

    print("[Coordinator] Waiting for threads to finish...")
    for t in threads:
        t.join()

    print("[Coordinator] Releasing camera...")
    camera.release()

    print("[Coordinator] All threads stopped. Exiting cleanly.")
