import cv2
import mediapipe as mp
import time
from ultis.hand_recognition import recognize_hand_gesture
from ultis.pose_recognition import recognize_pose
import threading
from ultis.face_function import initialize_faiss, recognize_face_realtime, draw_bbox
import os 

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(static_image_mode=False, model_complexity=1, smooth_landmarks=True, enable_segmentation=False)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

i = 0
warmup_frame = 60
recognition_thread = None
gesture_recognition_active = False  
gesture_start_time = 0
lock = threading.Lock() 
recognized_face = None
bbox = None
result = {"recognized_face": None, "bbox": None, "gesture_start_time": None,  "gesture_recognition_active": False}

initialize_faiss()

def recognition_task():
    global result
    face_result = recognize_face_realtime(rgb_frame)
    if face_result:
        recognized_face, bbox, gesture_start_time, gesture_recognition_active = face_result
        with lock:
            result.update({
                "recognized_face": recognized_face,
                "bbox": bbox,
                "gesture_start_time": gesture_start_time,
                "gesture_recognition_active": gesture_recognition_active
            })
    else:
        with lock:
            result.update({
                "recognized_face": None,
                "bbox": None,
                "gesture_start_time": None,
                "gesture_recognition_active": False
            })


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    i += 1  
    if i > warmup_frame:
        if recognition_thread is None or not recognition_thread.is_alive():
            recognition_thread = threading.Thread(target=recognition_task)
            recognition_thread.start()
        with lock:
            if result["recognized_face"]:
                draw_bbox(frame, result["recognized_face"], result["bbox"]) 
                start_time = time.time()
                if result["gesture_recognition_active"]:
                    
                    if time.time() - result["gesture_start_time"] < 15:
                        results = holistic.process(rgb_frame)
                        
                        p_landmarks = []
                        h_landmarks = [] 
        
                        if results.pose_landmarks:
                            for landmark in results.pose_landmarks.landmark:
                                p_landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
                                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

                        left_hand = [(lm.x, lm.y, lm.z) for lm in results.left_hand_landmarks.landmark] if results.left_hand_landmarks else [(0, 0, 0)] * 21
                        right_hand = [(lm.x, lm.y, lm.z) for lm in results.right_hand_landmarks.landmark] if results.right_hand_landmarks else [(0, 0, 0)] * 21
                        combined_hands = left_hand + right_hand

                        if results.left_hand_landmarks:
                            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                        if results.right_hand_landmarks:
                            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                        hand_gesture, c_score = recognize_hand_gesture(combined_hands)
        
                        cv2.putText(frame, f'Pose: {hand_gesture} ({c_score:.2f})', (200, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                        if not p_landmarks:
                            continue
                        gesture, confidence_score  = recognize_pose(p_landmarks,confidence_threshold = 0.85)
                        if gesture != 'Unidentified Pose':
                            cv2.putText(frame, f'Pose: {gesture} ({confidence_score:.2f})', (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                        else: 
                            cv2.putText(frame, 'Unidentified Pose', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                elapsed_time  = time.time() - start_time
                if elapsed_time > 0 :
                    fps = 1 / elapsed_time
                    cv2.putText(frame, f"fps: {fps:.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Recognition", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        cv2.putText(frame,'Warming up', (10, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

cap.release()
cv2.destroyAllWindows()
holistic.close()
