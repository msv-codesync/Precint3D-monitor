import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from paths import default_model_path

model = YOLO(str(default_model_path()))

# IP Camera app on the phone. Replace the host.
IP_CAMERA_URL = "http://192.168.1.XXX:8080/video"


def main():
    print(f"Connecting to iPhone camera at: {IP_CAMERA_URL}")
    cap = cv2.VideoCapture(IP_CAMERA_URL)
    if not cap.isOpened():
        print("Could not connect. Check the URL and that the phone app is open.")
        return

    print("Connected. Press q to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Lost connection to camera.")
            break
        results = model(frame, conf=0.40, verbose=False)
        annotated = results[0].plot()
        cv2.imshow("iPhone live — Precint3D", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
