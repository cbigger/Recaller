# camera_stream.py

import cv2
import argparse
import sys

class CameraStream:
    def __init__(self, cam_index=None, max_devices=10):
        self.cam_index = cam_index if cam_index is not None else self._prompt_for_camera(max_devices)
        self.cap = cv2.VideoCapture(self.cam_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.cam_index}")

    @staticmethod
    def list_available_cameras(max_devices=10):
        available = []
        for i in range(max_devices):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available

    def _prompt_for_camera(self, max_devices):
        cameras = self.list_available_cameras(max_devices)
        if not cameras:
            raise RuntimeError("No available cameras found.")

        print("Available cameras:")
        for i in cameras:
            print(f"[{i}] Camera {i}")

        while True:
            try:
                choice = int(input("Select camera index: "))
                if choice in cameras:
                    return choice
            except ValueError:
                pass
            print("Invalid selection.")

    def read_frame(self):
        if not self.cap.isOpened():
            raise RuntimeError("Camera is not opened.")
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from camera.")
        return frame

    def release(self):
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

    def stream(self, window_name="Camera Stream"):
        print(f"Streaming from camera {self.cam_index}. Press 'q' to quit.")
        while True:
            frame = self.read_frame()
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.release()

def main():
    parser = argparse.ArgumentParser(description="Test CameraStream module")
    parser.add_argument("--cam", type=int, help="Camera index to use")
    args = parser.parse_args()

    try:
        stream = CameraStream(cam_index=args.cam)
        stream.stream()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
