import os
import numpy as np
import pickle
import cv2
import faiss
import tensorflow as tf
import mediapipe as mp
import time 
from ultis.augmentation import augment_images
index_file = "faiss_index.bin"
user_map_file = "user_map.pkl"
d = 512  

index = None  
user_map = {} 
user_id_counter = 0 

interpreter = tf.lite.Interpreter(model_path='model\model_facenet_512.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def normalize_vector(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def preprocess_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(image_rgb, (160, 160))  
    normalized_img = resized_img / 255.0 
    return np.expand_dims(normalized_img.astype(np.float32), axis=0)

def get_embedding(face_crop):
    input_data = preprocess_image(face_crop)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    embedding = interpreter.get_tensor(output_details[0]['index'])[0]
    return normalize_vector(embedding)

def initialize_faiss():
    global index, user_map, user_id_counter

    if os.path.exists(index_file) and os.path.exists(user_map_file):
        index = faiss.read_index(index_file)
        print("FAISS index loaded from disk.")

        with open(user_map_file, "rb") as f:
            user_map = pickle.load(f)
            user_id_counter = len(user_map)  
        print("User map loaded from disk.")
    else:
        index = faiss.IndexFlatIP(d)  
        print("New FAISS index created.")

def save_faiss():
    faiss.write_index(index, index_file)
    print("FAISS index saved to disk.")
    with open(user_map_file, "wb") as f:
        pickle.dump(user_map, f)
    print("User map saved to disk.")

def detect_and_crop_face(image_paths):
    face_crops = []
    for img_path in image_paths:
        image = cv2.imread(img_path)
        if image is None:
            print(f"Error loading image: {img_path}")
            continue
        with mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            detection_results = face_detection.process(img_rgb)

            if not detection_results.detections:
                continue

            largest_detection = max(detection_results.detections, key=lambda det: det.location_data.relative_bounding_box.width * det.location_data.relative_bounding_box.height)

            bbox = largest_detection.location_data.relative_bounding_box
            h, w, _ = image.shape
            x_min = max(int(bbox.xmin * w), 0)
            y_min = max(int(bbox.ymin * h), 0)
            box_width = int(bbox.width * w)
            box_height = int(bbox.height * h)

            x_min = min(x_min, w - 1)
            y_min = min(y_min, h - 1)
            box_width = min(box_width, w - x_min)

            face_crop = image[y_min:y_min + box_height, x_min:x_min + box_width]
            if face_crop.size == 0:
                continue
            face_crops.append(face_crop)
    return face_crops

def register_new_user(user_name):
    global index, user_map, user_id_counter

    print(f"Registering new user: {user_name}")
    image_paths = [input(f"Enter the path for image {i+1}: ").strip() for i in range(15)]

    face_crop = detect_and_crop_face(image_paths)

    augmented_images = augment_images(face_crop, num_augmented=5)  

    for i, img in enumerate(augmented_images):
        try:
            if img is None:
                print(f"Failed to read image from {i}. Skipping.")
                continue

            embedding = get_embedding(img)

            embedding = np.array(embedding).astype('float32').reshape(1, -1)

            embedding = normalize_vector(embedding)

            index.add(embedding)

            user_map[user_id_counter] = user_name

            user_id_counter += 1

        except Exception as e:
            print(f"Error processing augmented image {i+1}: {e}")

    save_faiss()
    print(f"Registration completed for {user_name}.")

def recognize_face_from_image(img_path, threshold=0.6):
    global index, user_map

    img = cv2.imread(img_path)
    if img is None:
        print(f"Image not found: {img_path}.")
        return None

    face_crop = detect_and_crop_face(img)
    if face_crop is None:
        print("No face detected.")
        return None

    embedding = get_embedding(face_crop).reshape(1, -1)
    if index.ntotal > 0:
        distances, indices = index.search(embedding, k=1)
        if distances[0][0] > threshold:
            user_id = indices[0][0]
            user_name = user_map.get(user_id, "Unknown")
            print(f"User recognized: {user_name} (score: {distances[0][0]:.2f})")
            return user_name
        else:
            print("Unknown user.")
    else:
        print("No users in the database.")

    return None

def draw_bbox(frame, label, bbox):
    h, w, _ = frame.shape
    box_width = int(bbox.width * w)
    box_height = int(bbox.height * h)
    x_min = int(bbox.xmin * w)
    y_min = int(bbox.ymin * h)
    
    if label == 'Unknown':
        color = (0,0,255)
    elif label == 'No users registered':
        color = (255,0,0)
    else: 
        color = (0,255,0)        
    cv2.rectangle(frame, (x_min, y_min), (x_min + box_width, y_min + box_height), color, 2)
    cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

def recognize_face_realtime(frame, threshold=0.5):
    global index, user_map, recognized_face, gesture_recognition_active, gesture_start_time

    with mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:

        detection_results = face_detection.process(frame)

        if not detection_results.detections:
            return None, None, None

        for detection in detection_results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x_min = int(bbox.xmin * w)
            y_min = int(bbox.ymin * h)
            box_width = int(bbox.width * w)
            box_height = int(bbox.height * h)

            face_crop = frame[y_min:y_min + box_height, x_min:x_min + box_width]
            if face_crop.size == 0:
                print("Failed to crop the face region.")
                continue

            try:
                embedding = get_embedding(face_crop)
                embedding = np.array(embedding).astype('float32').reshape(1, -1)
                embedding = normalize_vector(embedding)

                if index.ntotal > 0:
                    distances, indices = index.search(embedding, k=1)

                    if distances[0][0] > threshold:
                        user_id = indices[0][0]
                        recognized_face = user_map.get(user_id, "Unknown")
                        gesture_recognition_active = True
                        gesture_start_time = time.time()   
                    else:
                        gesture_start_time = None
                        recognized_face = None
                        gesture_recognition_active = False  
                else:
                    gesture_start_time = None
                    recognized_face = None
                    gesture_recognition_active = False

                return recognized_face, bbox, gesture_start_time, gesture_recognition_active
                    
            except Exception as e:
                print(f"Error during embedding extraction or FAISS search: {e}")
                return None, None, None, None

    