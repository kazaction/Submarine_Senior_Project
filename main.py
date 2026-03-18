import onnxruntime
import onnx
import cv2
import numpy

filename = "./runs/detect/train14/weights/best.onnx"
onnx_model = onnx.load(filename)
cam = cv2.VideoCapture(0)
onnxruntime.InferenceSession(filename)

while True:
    #print("read")
    ret, frame = cam.read()
    # Display the frame with results
    cv2.imshow('Submarine Camera Feed', frame)

    #q button exits camera
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    #destroy resources allocated
    if not ret:
        cam.release()
        cv2.destroyAllWindows()
        break



#run(filename)

    input_size = (640, 640)
    frame_resized = cv2.resize(frame, input_size)
    input_tensor = numpy.array(frame_resized).astype(numpy.float32)
    input_tensor = numpy.expand_dims(input_tensor, axis=0)


    # Run inference
    #outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: input_tensor})


# Process the outputs as needed



