import cv2
import matplotlib.pyplot as plt
from ultralytics import solutions, YOLO

cap = cv2.VideoCapture("data/cars.mp4")
assert cap.isOpened(), "Error in reading video file"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,
                                       cv2.CAP_PROP_FRAME_HEIGHT,
                                       cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("speed_estimation.avi",
                               cv2.VideoWriter_fourcc(*"mp4v"),
                               fps, (w, h))

line_pts = [(0, int(h/2)), (w, int(h/2))]

# Initialize SpeedEstimator
speed_obj = solutions.SpeedEstimator(
    region=line_pts,
    model="models/yolo11n.pt",
    show=True,
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break
    im0 = speed_obj.estimate_speed(im0)
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()