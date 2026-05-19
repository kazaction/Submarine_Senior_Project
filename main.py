# import YOLO class from ultralytics library
# provides the interface for loading and
# running the trained model
from ultralytics import YOLO

# import OpenCV library for camera
# initialisation and frame capture
import cv2

# import sleep function froom time library
# to control the timings on the actuator
from time import sleep

# import the RPi.GPIOlibrary to control the GPIO pins
# on the Raspberry Pi, used to trigger the servo
# motor that operates the actuator
import RPi.GPIO as GPIO

# configuration

# path to the trained model weights exported to ONNX format
model_path = "./runs/detect/train18/weights/best.onnx"

# detection confidence threshold
# if a detection is below this threshold then it will not
# trigger the actuator. Set to 0.7 to favor precision over
# recall, minimising false positives
confidence_threshold = 0.7

# GPIO pin number connected to the servo motor signal wire
servo_pin = 18

# camera index. If multiple cameras are connected to a
# device this value might need to be changed to 1 or higher
camera_index = 0

# initialisations

# load the trained YOLO model from the specified ONNX file
# the ultralytics YOLO class automatically detects the ONNX
# format and uses the ONNX runtime on for inference
model = YOLO(model_path)

# initialise the camera capture object using the
# specified camera index. OpenCV handles the camera
# interface and provides raw frames for processing
cap = cv2.VideoCapture(camera_index)

# set the GPIO pin numbering to BCM
GPIO.setmode(GPIO.BCM)

# configure the servo pin as an output pin
GPIO.setup(servo_pin, GPIO.OUT)

# initialises a PWM instance on the servo pin at 50hz
# (this is the standard signal frequency for servo motors)
servo = GPIO.PWM(servo_pin, 50)
servo.start(0)

# main loop
# the main loop runs continuously (assuming that
# everything works as expected) until the user presses
# the key 'q' to quit
# Each iteration captures a single
# frame from the camera, runs the infernce on it and
# then processes and acts on the results
while True:
    # capture a frame from the camera
    # ret is a boolean indeicating whether the capture
    # was succesful
    # frame contains the raw image array.
    ret, frame = cap.read()

    # if the frame capture fails exit the loop here
    if not ret:
        break

    # run inference on the current frame
    # imgsz=640 resizes the frame to 640x640 pixels before
    # passing it to the model, matching the input
    # resolution used in training
    # conf applies the confidence threshold, discarding
    # any detections below this value before  returning any results
    results = model(frame, imgsz=640, conf=confidence_threshold)

    # check if any detections were returned by the model
    # if at least one detection was found issue a pwm signal
    # to the servo to deploy the spear actuator forward,
    # wait 0.5 seconds, retract the spear, and then wait again
    # 0.5 seconds
    if len(results[0].boxes) > 0:
        servo.ChangeDutyCycle(7.5)
        sleep(0.5)
        servo.ChangeDutyCycle(2.5)
        sleep(0.5)

    # generates an annotated version of the frame
    # with bounding boxes, labels and confidence scores
    annotated = results[0].plot()

    # display the annotated frame in a window for monitoring purposes
    cv2.imshow("Submarine Camera Feed", annotated)

    # wait 1 millisecon for every key event and
    # check if 'q' is pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release camera resource
cap.release()

# close all openCV windows
cv2.destroyAllWindows()

# stop the PWM signal and cleanup the GPIO configuration
servo.stop()
GPIO.cleanup()