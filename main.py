from ultralytics import YOLO
import cv2

model = YOLO("./runs/detect/train14/weights/best.onnx")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, imgsz=640, conf=0.35)

    annotated = results[0].plot()

    cv2.imshow("Submarine Camera Feed", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()