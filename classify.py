import cv2
import numpy as np
import tensorflow as tf

# Load waste classifier model
model = tf.keras.models.load_model('model/waste_classifier.h5')
class_names = ['biodegradable', 'non-biodegradable']  # class_indices assumed 0: biodegradable, 1: non-biodegradable

# YOLOv3 Setup
yolo_cfg = 'yolov3/yolov3.cfg'
yolo_weights = 'yolov3/yolov3.weights'
yolo_names = 'yolov3/coco.names'

# Load YOLO model
net = cv2.dnn.readNetFromDarknet(yolo_cfg, yolo_weights)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

def preprocess_frame(frame):
    img = cv2.resize(frame, (150, 150))  # Resize to match model input
    img = img / 255.0                    # Normalize
    img = np.expand_dims(img, axis=0)   # Add batch dimension
    return img

def classify_frame(frame):
    processed = preprocess_frame(frame)
    prediction = model.predict(processed)[0][0]
    class_index = int(round(prediction))  # 0 or 1
    label = class_names[class_index]
    confidence = prediction if class_index == 1 else 1 - prediction
    return label, confidence

def detect_objects(frame):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w // 2
                y = center_y - h // 2
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return indices, boxes, class_ids, confidences

def capture_and_classify():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return

    print("✅ Webcam is working. Press 'ESC' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        indices, boxes, class_ids, confidences = detect_objects(frame)

        for i in indices:
            i = i[0]
            box = boxes[i]
            x, y, w, h = box
            label, confidence = classify_frame(frame[y:y+h, x:x+w])
            color = (0, 255, 0) if label == 'biodegradable' else (0, 0, 255)
            text = f"{label} ({confidence*100:.2f}%)"
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Waste Classifier with Object Detection", frame)

        if cv2.waitKey(1) == 27:  # ESC key to break
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_classify()
