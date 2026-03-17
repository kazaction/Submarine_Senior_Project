import onnxruntime
import onnx
import cv2
import numpy as np
filename = "./runs/detect/train14/weights/best.onnx"
onnx_model = onnx.load(filename)
cam = cv2.VideoCapture(0)
onnxruntime.InferenceSession(filename)
while True:
    print("read")
    ret, frame = cam.read()
    # Display the frame with results
    cv2.imshow('Camera Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    
    if not ret: # Release resources
        cam.release()
        cv2.destroyAllWindows()
        break



#run(filename)

    input_size = (224, 224)  # Example size, adjust as needed
    frame_resized = cv2.resize(frame, input_size)
    input_tensor = np.array(frame_resized).astype(np.float32)
    input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dimension


    # Run inference
    #outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: input_tensor})


# Process the outputs as needed



