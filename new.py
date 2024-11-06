import mediapipe as mp
import cv2
import numpy as np
import csv 
import os 

cap = cv2.VideoCapture('C:\\Users\\truon\\OneDrive\\Pictures\\Camera Roll\\val_None_left.mp4')
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

data_dir = 'test_data'
current_gesture = 'val_None_left'

def save_landmarks(landmarks, data_dir):
    fileName = f"{current_gesture}.csv"
    filePath = os.path.join(data_dir, fileName)
    flattened_landmarks = np.array(landmarks).flatten()

    with open(filePath, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(flattened_landmarks)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgbFrame)

    left_hand_landmarks = [[0, 0, 0] for _ in range(21)]
    right_hand_landmarks = [[0, 0, 0] for _ in range(21)]
    
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):

            landmarks = [[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark]

            # if hand_handedness.classification[0].label == "Left":
            #     left_hand_landmarks = landmarks
            # else:
            #     right_hand_landmarks = landmarks

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # all_landmarks = left_hand_landmarks + right_hand_landmarks
        save_landmarks(landmarks, data_dir)

    cv2.putText(frame, f"Collecting: {current_gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Collect OK data', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()

# import csv
# import numpy as np
# import os

# def filter_rows_with_both_hands(input_file, output_file):
#     with open(input_file, 'r') as csvfile:
#         reader = csv.reader(csvfile)
#         valid_rows = []

#         for row in reader:

#             row_data = np.array(row, dtype=float)

#         #     left_hand = row_data[:63]
#         #     right_hand = row_data[63:]

#         #     if np.any(left_hand) and np.any(right_hand): 
#         #         valid_rows.append(row)
#             if np.any(row_data):
#                 valid_rows.append(row)
#     with open(output_file, 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerows(valid_rows)

# input_file = 'test_data\\val_OK.csv'
# output_file = 'test_data\\val_OK_filtered.csv'
# filter_rows_with_both_hands(input_file, output_file)
