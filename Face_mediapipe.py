import os
import numpy as np
import pickle
import cv2
import faiss
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultis.augmentation import augment_images 
import time
index_file = "faiss_index.bin"
user_map_file = "user_map.pkl"
d = 1024

index = None  
user_map = {} 
user_id_counter = 0 

def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def initialize_faiss():
    global index, user_map, user_id_counter

    if os.path.exists(index_file) and os.path.exists(user_map_file):
        index = faiss.read_index(index_file)

        with open(user_map_file, "rb") as f:
            user_map = pickle.load(f)
            user_id_counter = len(user_map) 
        index = faiss.IndexFlatIP(d)

def save_faiss():
    global index, user_map

    faiss.write_index(index, index_file)

    with open(user_map_file, "wb") as f:
        pickle.dump(user_map, f)

def load_image_embedder():
    base_options = python.BaseOptions(model_asset_path='facenet512.tflite')
    embedder_options = vision.ImageEmbedderOptions(base_options=base_options)
    image_embedder = vision.ImageEmbedder.create_from_options(embedder_options)
    return image_embedder

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
def get_embedding(face_crop, image_embedder):
    image_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

    image_resized = cv2.resize(image_rgb, (224, 224)) 

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_resized)

    embedding_result = image_embedder.embed(mp_image)
    
    if not embedding_result.embeddings:
        raise ValueError("No embeddings returned by the ImageEmbedder.")

    embedding_obj = embedding_result.embeddings[0]

    try:
        float_embedding = embedding_obj.embedding  
    except AttributeError:
        try:
            float_embedding = embedding_obj.float_values 
        except AttributeError:
            raise AttributeError("Unable to extract float values from the Embedding object.")
    
    return float_embedding

def register_new_user(user_name, image_embedder):
    global index, user_map, user_id_counter

    print(f"Registering new user: {user_name}")
    print("Please provide 15 images for registration.")

    image_paths = []
    for i in range(5):
        img_path = input(f"Enter the path for image {i+1}: ").strip()
        if not os.path.exists(img_path):
            print(f"Image {i+1} not found at {img_path}. Skipping.")
        else:
            image_paths.append(img_path)

    if not image_paths:
        print("No valid images provided for registration.")
        return
    
    face_crop = detect_and_crop_face(image_paths)

    augmented_images = augment_images(face_crop, num_augmented=3)  

    for i, img in enumerate(augmented_images):
        try:
            if img is None:
                print(f"Failed to read image from {i}. Skipping.")
                continue

            embedding = get_embedding(img, image_embedder)

            embedding = np.array(embedding).astype('float32').reshape(1, -1)

            embedding = normalize_vector(embedding)

            index.add(embedding)

            user_map[user_id_counter] = user_name
            print(f"Embedding saved for augmented image {i+1}. User ID: {user_id_counter}")

            user_id_counter += 1

        except Exception as e:
            print(f"Error processing augmented image {i+1}: {e}")

    save_faiss()
    print(f"Registration completed for {user_name}.")

def recognize_face_from_image(img_path, image_embedder, threshold=0.5):
    global index, user_map

    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image from {img_path}.")
            return None

        with mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            detection_results = face_detection.process(img_rgb)

            if not detection_results.detections:
                print("No face detected in the image.")
                return None
            
            detection = detection_results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = img.shape
            x_min = int(bbox.xmin * w)
            y_min = int(bbox.ymin * h)
            box_width = int(bbox.width * w)
            box_height = int(bbox.height * h)

            face_crop = img[y_min:y_min + box_height, x_min:x_min + box_width]
            if face_crop.size == 0:
                print("Failed to crop the face region.")
                return None

            embedding = get_embedding(face_crop, image_embedder)
            embedding = np.array(embedding).astype('float32').reshape(1, -1)
            embedding = normalize_vector(embedding)

            if index.ntotal > 0:
                distances, indices = index.search(embedding, k=1) 

                if distances[0][0] > threshold:
                    user_id = indices[0][0]
                    user_name = user_map.get(user_id, "Unknown")
                    print(f"User recognized as: {user_name} (similarity score: {distances[0][0]:.4f})")
                    return user_name
                else:
                    print("Unknown")
                    user_name = input("Enter the name for registration: ").strip()
                    register_new_user(user_name, image_embedder)
            else:
                print("No users registered in the database.")
                user_name = input("Enter the name for registration: ").strip()
                register_new_user(user_name, image_embedder)
    except Exception as e:
        print(f"Error in recognizing face: {e}")

    return None

def detect_face_and_handle_new(image_embedder):
    global index, user_map, user_id_counter

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam.")
        return

    with mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from webcam.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detection_results = face_detection.process(frame_rgb)

            if not detection_results.detections:
                cv2.imshow('Face Recognition', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            print("Face detected. Attempting to recognize...")
            user_name = recognize_face_from_image(frame, image_embedder)

            if user_name:
                print(f"Welcome back, {user_name}!")
            else:
                print("User not recognized. Initiating registration.")
                user_name = input("Enter your name for registration: ").strip()
                register_new_user(user_name, image_embedder)

            cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            break 

    cap.release()
    cv2.destroyAllWindows()

def recognize_face_realtime(image_embedder, threshold=0.75):
    global index, user_map

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam.")
        return

    with mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        prev_time = 0  # Initialize previous time

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from webcam.")
                break

            current_time = time.time()  

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detection_results = face_detection.process(frame_rgb)

            if not detection_results.detections:
                cv2.imshow('Face Recognition', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

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
                    embedding = get_embedding(face_crop, image_embedder)
                    embedding = np.array(embedding).astype('float32').reshape(1, -1)
                    embedding = normalize_vector(embedding)

                    if index.ntotal > 0:
                        distances, indices = index.search(embedding, k=1)

                        if distances[0][0] > threshold:
                            user_id = indices[0][0]
                            user_name = user_map.get(user_id, "Unknown")
                            label = f"{user_name} ({distances[0][0]:.4f})"
                            color = (0, 255, 0)  # Green for recognized
                            # print(f"User recognized as: {user_name} (similarity score: {distances[0][0]:.4f})")
                        else:
                            label = "Unknown"
                            color = (0, 0, 255)  # Red for unrecognized
                    else:
                        label = "No users registered"
                        color = (255, 0, 0)  # Blue

                    cv2.rectangle(frame, (x_min, y_min), (x_min + box_width, y_min + box_height), color, 2)
                    cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                except Exception as e:
                    print(f"Error during embedding extraction or FAISS search: {e}")
            
            elapsed_time = time.time() - current_time  
            fps = 1 / elapsed_time  
            cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# Main flow
def main():
    global index, user_map, user_id_counter

    initialize_faiss()

    image_embedder = load_image_embedder()

    print("Select mode:")
    print("1. Real-time Face Recognition")
    print("2. Register New User")
    print("2. Pic Face Recognition")
    mode = input("Choose mode from 1 to 3: ").strip()

    if mode == '1':
        recognize_face_realtime(image_embedder, threshold=0.5) 
    elif mode == '2':
        user_name = input("Enter the name for registration: ").strip()
        register_new_user(user_name, image_embedder)
    elif mode == '3':
        img_path = input('Enter img path for recognize: ').strip()
        recognize_face_from_image(img_path, image_embedder, threshold=0.6)
    else:
        print("Invalid mode selected.")

if __name__ == "__main__":
    main()
