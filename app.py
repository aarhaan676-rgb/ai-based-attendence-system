
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace
from datetime import datetime
import os
import tempfile

st.set_page_config(page_title="AI Attendance System", layout="centered")
st.title("AI Based Attendance System")

attendance_file = "attendance.csv"
embeddings_file = "embeddings.pkl"
dataset_path = "dataset"  # relative path

# Utility: Find best match
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

# Utility: Mark attendance
def mark_attendance(name):
    if not os.path.exists(attendance_file):
        pd.DataFrame(columns=["Name", "Time"]).to_csv(attendance_file, index=False)
    attendance = pd.read_csv(attendance_file)
    today = datetime.now().strftime("%Y-%m-%d")
    # Check if already marked for today
    already_marked = False
    for idx, row in attendance.iterrows():
        if str(row["Name"]).strip() == str(name).strip() and str(row["Time"]).startswith(today):
            already_marked = True
            break
    if not already_marked:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        attendance.loc[len(attendance)] = [name, now]
        attendance.to_csv(attendance_file, index=False)

st.sidebar.header("Options")
option = st.sidebar.radio("Choose Mode", ["Train Embeddings", "Recognise from Webcam", "Recognise from Image", "View Attendance"])

if option == "Train Embeddings":
    st.subheader("Train Embeddings from Dataset")
    if st.button("Start Training"):
        records = []
        for person in os.listdir(dataset_path):
            person_folder = os.path.join(dataset_path, person)
            if not os.path.isdir(person_folder):
                continue
            for img_name in os.listdir(person_folder):
                img_path = os.path.join(person_folder, img_name)
                try:
                    embedding = DeepFace.represent(img_path=img_path, model_name="Facenet")[0]["embedding"]
                    records.append({
                        "name": person,
                        "img_path": img_path,
                        "embedding": embedding
                    })
                    st.write(f"Embedded: {img_path}")
                except Exception as e:
                    st.warning(f"Error with {img_path}: {e}")
        if records:
            df = pd.DataFrame(records)
            df.to_pickle(embeddings_file)
            st.success("Training Complete! Embeddings saved.")
        else:
            st.error("No embeddings generated. Check your dataset.")

elif option == "Recognise from Webcam":
    st.subheader("Recognise Face from Webcam")
    st.info("Click 'Start Webcam' and allow access. Press 'Q' in the webcam window to quit.")
    if st.button("Start Webcam"):
        if not os.path.exists(embeddings_file):
            st.error("Embeddings not found. Please train first.")
        else:
            df = pd.read_pickle(embeddings_file)
            # Start webcam
            cap = cv2.VideoCapture(0)
            st.write("Press Q to quit webcam window.")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                try:
                    result = DeepFace.represent(frame, model_name="Facenet")[0]["embedding"]
                    name = find_match(result, df)
                    if name and str(name).strip():
                        mark_attendance(name)
                        cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "No Face", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                except Exception as e:
                    cv2.putText(frame, "No Face", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Attendance System - DeepFace", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
            st.success("Webcam session ended.")

elif option == "Recognise from Image":
    st.subheader("Recognise Face from Uploaded Image")
    uploaded = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])
    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded.read())
        if not os.path.exists(embeddings_file):
            st.error("Embeddings not found. Please train first.")
        else:
            df = pd.read_pickle(embeddings_file)
            try:
                test_embedding = DeepFace.represent(img_path=tfile.name, model_name="Facenet")[0]["embedding"]
                name = find_match(test_embedding, df)
                st.image(uploaded, caption=f"Recognized: {name}")
                if name and str(name).strip():
                    mark_attendance(name)
                    st.success(f"Attendance Marked for {name}")
                else:
                    st.warning("No face recognized. Attendance not marked.")
            except Exception as e:
                st.error("Face not detected.")
                st.write(e)

elif option == "View Attendance":
    st.subheader("Attendance Records")
    if os.path.exists(attendance_file):
        attendance = pd.read_csv(attendance_file)
        st.dataframe(attendance)
    else:
        st.info("No attendance records found.")
