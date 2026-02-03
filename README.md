Current dependencies:
pip install torch torchvision
pip install ultralytics
yolo detect train model=yolov8n.pt data=lionfish.yaml epochs=50 imgsz=640 batch=16
yolo detect predict model=C:\Users\35797\PycharmProjects\Submarine_Senior_Project\runs\detect\train10\weights\best.pt source=0 conf=0.4 show=True

