import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace
from datetime import datetime

# Load embedding database
df = pd.read_pickle("embeddings.pkl")

def find_match(embedding, df, threshold=0.90):
    best_match = None
    lowest_dist = 10
    embedding = np.array(embedding)
    for i, row in df.iterrows():
        db_embedding = np.array(row["embedding"])
        dist = np.linalg.norm(embedding - db_embedding)
        if dist < lowest_dist:
            lowest_dist = dist
            best_match = row["name"]
    return best_match


attendance_file = "attendance.csv"

# Initialize CSV
import os
if not os.path.exists(attendance_file):
    pd.DataFrame(columns=["Name", "Time"]).to_csv(attendance_file, index=False)


# --- Face Detection Model Setup ---
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
faceNet = cv2.dnn.readNet(faceModel, faceProto)

def highlightFace(net, frame, conf_threshold=0.8):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            # Add more padding
            pad = 40
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(frameWidth - 1, x2 + pad)
            y2 = min(frameHeight - 1, y2 + pad)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, faceBoxes

# Start webcam
cap = cv2.VideoCapture(0)

print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resultImg, faceBoxes = highlightFace(faceNet, frame)
    name = "No Face"
    if not faceBoxes:
        cv2.putText(resultImg, name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print("No face detected")
    else:
        for faceBox in faceBoxes:
            # Crop face for DeepFace
            face = frame[max(0, faceBox[1]):min(faceBox[3], frame.shape[0]-1),
                        max(0, faceBox[0]):min(faceBox[2], frame.shape[1]-1)]
            print(f"Cropped face shape: {face.shape}")
            # Skip faces smaller than 80x80 pixels
            if face.size == 0 or face.shape[0] < 80 or face.shape[1] < 80:
                cv2.putText(resultImg, "Invalid Face", (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                print("Face region too small or empty, skipping.")
                continue
            try:
                rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                result = DeepFace.represent(rgb_face, model_name="Facenet")[0]["embedding"]
                name = find_match(result, df)
                # Mark Attendance only once per person per day, skip empty names
                if name and str(name).strip():
                    attendance = pd.read_csv(attendance_file)
                    today = datetime.now().strftime("%Y-%m-%d")
                    already_marked = False
                    for idx, row in attendance.iterrows():
                        if str(row["Name"]).strip() == str(name).strip() and str(row["Time"]).startswith(today):
                            already_marked = True
                            break
                    if not already_marked:
                        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        attendance.loc[len(attendance)] = [name, now]
                        attendance.to_csv(attendance_file, index=False)
                        cv2.putText(resultImg, name, (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow("Attendance System - DeepFace", resultImg)
                        print(f"Attendance marked for {name} at {now}")
                        cap.release()
                        cv2.destroyAllWindows()
                        exit(0)
                    else:
                        cv2.putText(resultImg, name + " (Already marked)", (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            except Exception as e:
                cv2.putText(resultImg, "No Face", (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                print(f"Face detection error: {e}")

    cv2.imshow("Attendance System - DeepFace", resultImg)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()