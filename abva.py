# import os
# import cv2
# import json

# images_dir = 'D:\\ToaDA\\hagrid-sample-30k-384p\\hagrid_30k\\train_val_three2'
# annotations_dir = 'D:\\ToaDA\\hagrid-sample-30k-384p\\ann_train_val'
# output_dir = 'D:\ToaDA\hagrid-sample-30k-384p\hagrid_30k\crop_three2'

# os.makedirs(output_dir, exist_ok=True)

# for json_file in os.listdir(annotations_dir):
#     if json_file.endswith('.json'):

#         with open(os.path.join(annotations_dir, json_file)) as f:
#             data = json.load(f)
        
#         for image_id, details in data.items():
#             labels = details.get('labels', [])
#             bboxes = details.get('bboxes', [])

#             if "three2" in labels:

#                 dislike_index = labels.index("three2")
#                 bbox = bboxes[dislike_index]
                
#                 # Load image
#                 image_path = os.path.join(images_dir, f"{image_id}.jpg")
#                 image = cv2.imread(image_path)
#                 if image is None:
#                     continue  # Skip if image doesn't exist
                
#                 # Scale bbox to image dimensions
#                 img_height, img_width = image.shape[:2]
#                 x, y, w, h = bbox
#                 x, y, w, h = int(x * img_width), int(y * img_height), int(w * img_width), int(h * img_height)
                
#                 # Crop the gesture
#                 cropped_gesture = image[y:y+h, x:x+w]
                
#                 # Save the cropped image
#                 output_path = os.path.join(output_dir, f"{image_id}_label.jpg")
#                 cv2.imwrite(output_path, cropped_gesture)

import mediapipe as mp
import cv2
import numpy as np
import csv 
import os 

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

data_dir = 'hand_gesture_data'
image_dir = 'D:\\ToaDA\\rps_data_sample\\none'  
current_gesture = 'none'

def save_landmarks(landmarks, data_dir):
    fileName = f"{current_gesture}.csv"
    filePath = os.path.join(data_dir, fileName)
    flattened_landmarks = np.array(landmarks).flatten()

    with open(filePath, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(flattened_landmarks)

for image_name in range(len(os.listdir(image_dir))):
    
        image_path = os.path.join(image_dir, f"None ({image_name}).jpg")
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Could not read image {image_name}")
            continue

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = [[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark]
                save_landmarks(landmarks, data_dir)
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # cv2.putText(image, f"Collecting: {current_gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # cv2.imshow('Collect Data', image)
        
        if cv2.waitKey(100) & 0xFF == ord('q'):  # Adjust wait time as needed
            break

cv2.destroyAllWindows()
hands.close()

# import os
# import random
# import shutil

# # Define the main directory containing the 16 image folders
# main_dir = "D:\ToaDA\hagrid-sample-30k-384p\crop"
# output_dir = "D:\ToaDA\hagrid-sample-30k-384p\data"  # Where to save the selected images
# os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

# # Set the number of images to sample from each folder
# num_images_to_sample = 300

# # Iterate over each folder in the main directory
# for folder_name in os.listdir(main_dir):
#     folder_path = os.path.join(main_dir, folder_name)

#     if os.path.isdir(folder_path):

#         images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

#         if len(images) >= num_images_to_sample:

#             selected_images = random.sample(images, num_images_to_sample)
            
#             for image in selected_images:
#                 src_path = os.path.join(folder_path, image)
#                 dest_path = os.path.join(output_dir, f"{folder_name}_{image}") 
#                 shutil.copy(src_path, dest_path)
            
#             print(f"Selected {num_images_to_sample} images from folder: {folder_name}")
#         else:
#             print(f"Not enough images in folder: {folder_name}")