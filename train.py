import os
import cv2
import numpy as np
from deepface import DeepFace
from datetime import datetime
import pandas as pd

db_path = "./dataset/"
# This list will store embeddings
records = []
for person in os.listdir(db_path):
    person_folder = os.path.join(db_path, person)

    if not os.path.isdir(person_folder):
        continue

    for img_name in os.listdir(person_folder):
        img_path = os.path.join(db_path, person, img_name)

        try:
            embedding = DeepFace.represent(img_path=img_path, model_name="Facenet")[0]["embedding"]

            records.append({
                "name": person,
                "img_path": img_path,
                "embedding": embedding
            })

            print(f"Embedded: {img_path}")

        except Exception as e:
            print("Error:", e)

df = pd.DataFrame(records)
df.to_pickle("embeddings.pkl")
print("Training Complete! Embeddings saved as embeddings.pkl")

