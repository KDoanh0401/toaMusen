import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import numpy as np
import pickle
import cv2
import faiss
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultis.augmentation import augment_images 
import tensorflow as tf

index_file = "faiss_index.bin"
user_map_file = "user_map.pkl"
d = 512

index = None  
user_map = {} 
user_id_counter = 0 

registration_window_open = False
registration_lock = threading.Lock()

interpreter = tf.lite.Interpreter(model_path='model\model_facenet_512.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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
    else:
        index = faiss.IndexFlatIP(d)

def save_faiss():
    global index, user_map

    faiss.write_index(index, index_file)

    with open(user_map_file, "wb") as f:
        pickle.dump(user_map, f)

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

            largest_detection = max(
                detection_results.detections,
                key=lambda det: det.location_data.relative_bounding_box.width * det.location_data.relative_bounding_box.height
            )

            bbox = largest_detection.location_data.relative_bounding_box
            h, w, _ = image.shape
            x_min = max(int(bbox.xmin * w), 0)
            y_min = max(int(bbox.ymin * h), 0)
            box_width = int(bbox.width * w)
            box_height = int(bbox.height * h)

            x_min = min(x_min, w - 1)
            y_min = min(y_min, h - 1)
            box_width = min(box_width, w - x_min)
            box_height = min(box_height, h - y_min)

            face_crop = image[y_min:y_min + box_height, x_min:x_min + box_width]
            if face_crop.size == 0:
                continue
            face_crops.append(face_crop)
    return face_crops

def register_new_user(user_name, image_paths):
    global index, user_map, user_id_counter

    print(f"Registering new user: {user_name}")
    print(f"Number of images provided: {len(image_paths)}")

    if not image_paths:
        print("No valid images provided for registration.")
        return

    face_crops = detect_and_crop_face(image_paths)

    if not face_crops:
        print("No faces detected in the provided images.")
        return

    augmented_images = augment_images(face_crops, num_augmented=20)

    for i, img in enumerate(augmented_images):
        try:
            if img is None:
                print(f"Failed to read image from augmented image {i}. Skipping.")
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

def recognize_face_from_image(img_path, threshold=0.5):
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

            embedding = get_embedding(face_crop)
            embedding = np.array(embedding).astype('float32').reshape(1, -1)
            embedding = normalize_vector(embedding)

            if index.ntotal > 0:
                distances, indices = index.search(embedding, k=1) 

                if distances[0][0] > threshold:
                    user_id = indices[0][0]
                    user_name = user_map.get(user_id, "Unknown")
                    return user_name
                else:
                    print("Unknown")
                    open_registration_window()
            else:
                print("No users registered in the database.")
                open_registration_window()
    except Exception as e:
        print(f"Error in recognizing face: {e}")

    return None

def open_registration_window():
    global registration_window_open

    with registration_lock:
        if registration_window_open:
            return
        registration_window_open = True

    def submit_registration():
        user_name = name_entry.get().strip()
        if not user_name:
            messagebox.showerror("Input Error", "Please enter a valid name.")
            return

        if not selected_files:
            messagebox.showerror("Input Error", "Please select at least one image.")
            return

        reg_window.destroy()

        register_new_user(user_name, selected_files)
        messagebox.showinfo("Registration", f"User '{user_name}' registered successfully.")

        with registration_lock:
            registration_window_open = False

    def browse_files():
        files = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
        )
        if files:
            selected_files.clear()
            selected_files.extend(files)
            files_label.config(text=f"{len(selected_files)} files selected.")

    selected_files = []

    reg_window = tk.Tk()
    reg_window.title("Register New User")
    reg_window.geometry("400x200")
    reg_window.resizable(False, False)

    name_label = tk.Label(reg_window, text="Enter Your Name:")
    name_label.pack(pady=10)

    name_entry = tk.Entry(reg_window, width=40)
    name_entry.pack(pady=5)

    browse_button = tk.Button(reg_window, text="Browse Images", command=browse_files)
    browse_button.pack(pady=10)

    files_label = tk.Label(reg_window, text="No files selected.")
    files_label.pack(pady=5)

    submit_button = tk.Button(reg_window, text="Register", command=submit_registration)
    submit_button.pack(pady=20)

    def on_closing():
        if messagebox.askokcancel("Quit", "Do you want to quit registration?"):
            reg_window.destroy()
            with registration_lock:
                registration_window_open = False

    reg_window.protocol("WM_DELETE_WINDOW", on_closing)

    reg_window.mainloop()

def recognize_face_realtime(threshold=0.75):
    global index, user_map

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
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
                            user_name = user_map.get(user_id, "Unknown")
                            label = f"{user_name} ({distances[0][0]:.2f})"
                            color = (0, 255, 0)
                        else:
                            label = "Unknown"
                            color = (0, 0, 255) 

                            with registration_lock:
                                if not registration_window_open:
                                    reg_thread = threading.Thread(target=open_registration_window)
                                    reg_thread.start()

                    else:
                        label = "No users registered"
                        color = (255, 0, 0) 

                        with registration_lock:
                            if not registration_window_open:
                                reg_thread = threading.Thread(target=open_registration_window)
                                reg_thread.start()

                    cv2.rectangle(frame, (x_min, y_min), (x_min + box_width, y_min + box_height), color, 2)
                    cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                except Exception as e:
                    print(f"Error during embedding extraction or FAISS search: {e}")

            cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

def main():
    global index, user_map, user_id_counter

    initialize_faiss()

    print("Select mode:")
    print("1. Real-time Face Recognition")
    print("2. Register New User")
    print("3. Pic Face Recognition")
    mode = input("Choose mode from 1 to 3: ").strip()

    if mode == '1':
        recognize_face_realtime(threshold=0.5)
    elif mode == '2':
        user_name = input("Enter the name for registration: ").strip()
        image_paths = []
        for i in range(5):
            img_path = input(f"Enter the path for image {i+1}: ").strip()
            if not os.path.exists(img_path):
                print(f"Image {i+1} not found at {img_path}. Skipping.")
            else:
                image_paths.append(img_path)
        register_new_user(user_name, image_paths)
    elif mode == '3':
        img_path = input('Enter img path for recognize: ').strip()
        recognize_face_from_image(img_path, threshold=0.6)
    else:
        print("Invalid mode selected.")

if __name__ == "__main__":
    main()
