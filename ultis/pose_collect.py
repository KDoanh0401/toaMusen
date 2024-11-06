import mediapipe as mp
import cv2
import os
import numpy as np
import csv

# Initialize MediaPipe Pose
mpPose = mp.solutions.pose
Pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

label = 'None'
cap = cv2.VideoCapture('None.mp4')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
data_dir = 'val_dataset'

# Create directory if it doesn't exist
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Function to save pose landmarks
def save_pose_landmarks(frame_id, landmarks, data_dir, label):
    fileName = f"{label}.csv"
    filePath = os.path.join(data_dir, fileName)

    flattened_landmarks = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks.landmark]).flatten()
    row = [frame_id] + flattened_landmarks.tolist()

    file_exists = os.path.isfile(filePath)

    with open(filePath, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            headers = ['frame']
            for i in range(33):  
                headers += [f"landmark_{i}_x", f"landmark_{i}_y", f"landmark_{i}_z", f"landmark_{i}_visibility"]
            writer.writerow(headers)

        writer.writerow(row)

frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = Pose.process(rgbFrame)

    if results.pose_landmarks:
        save_pose_landmarks(frame_id, results.pose_landmarks, data_dir, label)
        mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

    frame_id += 1
    cv2.imshow("Image", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
