import torch
import cv2
import mediapipe as mp
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
import torch.nn.functional as F
import os
from torch.nn import BatchNorm1d, Dropout

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
def normalize_row(row):
        data = np.array(row).reshape(3, 21)
        
        wrist_z = data[2, 0]
        data[2, :] = data[2, :] - wrist_z

        for axis in range(3):
            axis_min = data[axis, :].min()
            axis_max = data[axis, :].max()
            if axis_max - axis_min != 0:
                data[axis, :] = (data[axis, :] - axis_min) / (axis_max - axis_min)
            else:
                data[axis, :] = 0
        
        return data.T
class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 64)  
        self.bn1 = BatchNorm1d(64) 
        self.conv2 = GCNConv(64, 128)  
        self.bn2 = BatchNorm1d(128)
        self.conv3 = GCNConv(128, num_classes)  

        self.dropout = Dropout(0.3)  

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)  

        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = self.dropout(x) 
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch) 
        return F.log_softmax(x, dim=1)
    

model = GCN(num_features=3, num_classes=3)  
model.load_state_dict(torch.load('gcn_hand_gesture_model.pth'))
model.eval()


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,max_num_hands = 1, min_detection_confidence=0.6)
mpDraw = mp.solutions.drawing_utils

def create_edges():
    hand_connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),   
        (0, 5), (5, 6), (6, 7), (7, 8),   
        (0, 9), (9, 10), (10, 11), (11, 12),  
        (0, 13), (13, 14), (14, 15), (15, 16), 
        (0, 17), (17, 18), (18, 19), (19, 20),  
        (5, 9),  
        (9, 13),
        (13, 17) 
    ]

    edge_index = torch.tensor(hand_connections, dtype=torch.long).t().contiguous()
    return edge_index

edge_index = create_edges()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
i = 0
warmup_frame = 60
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    i += 1
    if i > warmup_frame:
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mpDraw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)


                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y, landmark.z])

                landmarks_df = normalize_row(landmarks)
                landmarks_df = torch.tensor(landmarks, dtype=torch.float)
                graph = Data(x=landmarks_df, edge_index=edge_index)

                with torch.no_grad():
                    output = model(graph)
                    probabilities = torch.exp(output)  
                    predicted_class = probabilities.argmax(dim=1).item()
                    confidence_score = probabilities.max(dim=1).values.item()

                if confidence_score > 0.5:
                    # gesture = 'OK' if predicted_class == 1 else 'None'
                    # cv2.putText(frame, f'Gesture: {gesture} ({confidence_score:.2f})', (10, 40), 
                    #         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    match predicted_class :
                        case 0: 
                            gesture = 'none'
                        case 1: 
                            gesture = 'rock'
                        case 2:
                            gesture = 'paper'
                    cv2.putText(frame, f'Gesture: {gesture} ({confidence_score:.2f})', (10, 40), 
                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Hand Gesture Recognition", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        cv2.putText(frame,'Warming up', (10, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

cap.release()
cv2.destroyAllWindows()
hands.close()

# import torch
# import cv2
# import mediapipe as mp
# import numpy as np
# from torch_geometric.data import Data
# from torch_geometric.nn import GCNConv, global_mean_pool
# import torch.nn.functional as F
# import os
# import pandas as pd
# import time 
# from torch.nn import BatchNorm1d, Dropout

# # GAT Model Definition
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # class GCN(torch.nn.Module):
# #     def __init__(self, num_features, num_classes):
# #         super(GCN, self).__init__()
# #         self.conv1 = GCNConv(num_features, 32)  
# #         self.conv2 = GCNConv(32, num_classes)    

# #     def forward(self, data):
# #         x, edge_index, batch = data.x, data.edge_index, data.batch
# #         x = F.relu(self.conv1(x, edge_index))
# #         x = self.conv2(x, edge_index)
# #         x = global_mean_pool(x, batch)
# #         return F.log_softmax(x, dim=1)

# class GCN(torch.nn.Module):
#     def __init__(self, num_features, num_classes):
#         super(GCN, self).__init__()
#         self.conv1 = GCNConv(num_features, 64)  
#         self.bn1 = BatchNorm1d(64) 
#         self.conv2 = GCNConv(64, 128)  
#         self.bn2 = BatchNorm1d(128)
#         self.conv3 = GCNConv(128, num_classes)  

#         self.dropout = Dropout(0.5)  

#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch

#         x = F.relu(self.bn1(self.conv1(x, edge_index)))
#         x = self.dropout(x)  

#         x = F.relu(self.bn2(self.conv2(x, edge_index)))
#         x = self.dropout(x) 
#         x = self.conv3(x, edge_index)
#         x = global_mean_pool(x, batch) 
#         return F.log_softmax(x, dim=1)
    
# model = GCN(num_features=4, num_classes=3).to(device) 
# model.load_state_dict(torch.load('gcn_pose_gesture_model.pth', map_location=device))
# model.eval()

# mpPose = mp.solutions.pose
# Pose = mpPose.Pose()
# mpDraw = mp.solutions.drawing_utils

# def create_edges():
#     body_connections = [
#     (11, 13), (13, 15), (15, 17), (17, 19), (15, 21),(15, 19),                              
#     (12, 14), (14, 16), (16, 18), (18, 20), (20, 16), (16,22),  
#     (11, 12)
#     ]

#     edge_index = torch.tensor(body_connections, dtype=torch.long).t().contiguous() 
#     return edge_index.to(device) 

# def center_data(landmarks):
#     data = np.array(landmarks).reshape(33, 4).T
#     for axis in range(3):  
#             data[axis, :] = data[axis, :] - (data[axis, 23] + data[axis, 24]) / 2

#     centered_row = data.T
#     return centered_row

# def normalize_data(landmarks):
#     data = np.array(landmarks).reshape(33, 4).T
#     for axis in range(3):  
#         axis_min = data[axis, :].min()
#         axis_max = data[axis, :].max()
#         if axis_max - axis_min != 0:
#             data[axis, :] = (data[axis, :] - axis_min) / (axis_max - axis_min)
#         else:
#             data[axis, :] = 0  

#     normalized_row = data.T
#     return normalized_row 

# edge_index = create_edges()

# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# i = 0

# warmup_frame = 60
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     i += 1
#     if i > warmup_frame:
#         start_time = time.time()
#         results = Pose.process(rgb_frame)

#         if results.pose_landmarks:
            
#             mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

            
#             landmarks = []
#             for landmark in results.pose_landmarks.landmark:
#                 landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])

#             landmarks_df = center_data(landmarks)
#             landmarks_df = normalize_data(landmarks_df)
#             landmarks_df = torch.tensor(landmarks_df, dtype=torch.float).to(device) 
         
#             graph = Data(x=landmarks_df, edge_index=edge_index)

                
#             with torch.no_grad():
#                 output = model(graph)
#                 probabilities = torch.exp(output)  
#                 predicted_class = probabilities.argmax(dim=1).item()
#                 confidence_score = probabilities.max(dim=1).values.item()

#             if confidence_score > 0.85:
#                 # gesture = 'OK' if predicted_class == 1 else 'NO'
#                 match predicted_class:
#                     case 0:
#                         gesture = 'NO'
#                     case 1:
#                         gesture = 'OK'
#                     case 2:
#                         gesture = 'Unidentified Pose'
#                 if gesture != 'Unidentified Pose':
#                     cv2.putText(frame, f'Pose: {gesture} ({confidence_score:.2f})', (10, 40), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
#                 else:
#                     cv2.putText(frame, 'Unidentified Pose', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
#             else: 
#                 cv2.putText(frame, 'Unidentified Pose', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
#         elapsed_time  = time.time() - start_time
#         fps = 1 / elapsed_time
#         if elapsed_time > 0 :
#             cv2.putText(frame, f"fps: {fps:.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

#         cv2.imshow("Pose Recognition", frame)
#         if cv2.waitKey(1) == ord('q'):
#             break
#     else:
#         cv2.putText(frame,'Warming up', (10, 40), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# cap.release()
# cv2.destroyAllWindows()
# Pose.close()




