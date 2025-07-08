import sys
import argparse
import threading
from queue import Queue
# Relative imports
import config
from targetter import FacePackager, FaceTrainer
from camera_stream import CameraStream
from producer import FrameProducer
from workers import FaceWorker, DetectWorker

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


def run_targetter(args, nms_mode=True):
    pass

# Will parse arguments and default to live-feed mode if targetting mode is not set.
def parse_args_with_default_mode(default_mode="live"):
    parser = argparse.ArgumentParser(
        description="Recaller: A python library that combines person detection and facial recognition to pick out a specific person from a crowd.\nUse --help in combination with your mode selection to get a list of mode-specific arguments"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

  # Subparser for default live feed mode
    parser_mode1 = subparsers.add_parser("live", help="Run in live mode; default behaviour, does not need to be set even if using additional sub-args.")
    parser_mode1.add_argument("--cam", type=int, help="Camera index to use. No default, but you probably want 0 unless you have multiple cams hooked up.")

  # Subparser for targetting mode
    parser_mode2 = subparsers.add_parser("targetter", help="Run the still image target packager.")
    parser_mode2.add_argument("--build_test_images", action="store_true", help="Export to off-target face dataset instead of target set.")
    parser_mode2.add_argument("--no_nms", action="store_true", help="Disable NMS during face packaging.")
    parser_mode2.add_argument("--test_recognition", action="store_true", help="Run a recognition test and print the scores to the console.")

    # If help is explicitly requested, show global help
    if "--help" in sys.argv or "-h" in sys.argv:
        parser.parse_args()

    # Otherwise insert default mode if no mode is provided
    if len(sys.argv) > 1 and sys.argv[1] not in ["live", "targetter"]:
        args = parser.parse_args([default_mode] + sys.argv[1:])
    elif len(sys.argv) == 1:
        args = parser.parse_args([default_mode])
    else:
        args = parser.parse_args()

    return args

def main():
    # Run the arg system to get the mode and arguments set properly
    args =  parse_args_with_default_mode() #parser.parse_args()
    # alright so at this moment we are trying to split up our entries
    # we should be handling all argument stuff here, and pass the
    # decoded arguments as parameters for our class.

    if args.mode == "targetter":
        # Set the defaults
        input_dir = config.INPUT_DIR
        output_dir = config.OUTPUT_DIR
        nms_mode = True

        # Override based on arguments
        if args.test_recognition:
            test_recognizer = FaceTrainer(output_dir, config.OFFTARGET_OUTPUT_DIR, config.TARGET_FACE_PATH)
            test_recognizer.train()
            test_recognizer.test()

        else:
            if args.build_test_images: # Populate the offtarget directory instead of normal one.
                input_dir = config.OFFTARGET_INPUT_DIR
                output_dir = config.OFFTARGET_OUTPUT_DIR
            elif args.no_nms:
                nms_mode = False

            target_packager = FacePackager(input_dir, output_dir, nms_mode=nms_mode)
            target_packager.run()

    else:

        run_system()

if __name__ == "__main__":
    main()
