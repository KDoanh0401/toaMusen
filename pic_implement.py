import cv2
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
import mediapipe as mp
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
import os
from torch.nn import BatchNorm1d, Dropout

# class GCN(torch.nn.Module):
#     def __init__(self, num_features, num_classes):
#         super(GCN, self).__init__()
#         self.conv1 = GCNConv(num_features, 32)  
#         self.conv2 = GCNConv(32, num_classes)   
#         self.dropout = torch.nn.Dropout(0.3)    
        
#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch

#         x = F.relu(self.conv1(x, edge_index))
#         x = self.dropout(x)
#         x = self.conv2(x, edge_index)

#         x = global_mean_pool(x, batch)
        
#         return F.log_softmax(x, dim=1) 
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

model = GCN(num_features=3, num_classes=2)
model_path = 'gcn_hand_gesture_model.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    print("GCN model loaded successfully.")
else:
    print(f"Model file {model_path} not found.")
    exit()

def create_edges():
    # hand_connections = [
    #     (0, 1), (1, 2), (2, 3), (3, 4),   
    #     (0, 5), (5, 6), (6, 7), (7, 8),   
    #     (0, 9), (9, 10), (10, 11), (11, 12),  
    #     (0, 13), (13, 14), (14, 15), (15, 16), 
    #     (0, 17), (17, 18), (18, 19), (19, 20),  
    #     (5, 9),  
    #     (9, 13),
    #     (13, 17) 
    # ]
    hand_connections = [
        (0,2), (2,4), 
        (0, 5), (5, 6), (6, 7), (7, 8),   
        (0, 9), (9, 10), (10, 11), (11, 12),  
        (0, 13), (13, 14), (14, 15), (15, 16), 
        (0, 17), (17, 18), (18, 19), (19, 20) 
    ]
    edge_index = torch.tensor(hand_connections, dtype=torch.long).t().contiguous()
    return edge_index

edge_index = create_edges()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

def recognize_gesture(landmarks):
    if len(landmarks) < 21:  
        return None, None

    try:
        node_features = torch.tensor(landmarks).reshape(-1, 3).float()  
        data = Data(x=node_features, edge_index=edge_index, batch=torch.tensor([0]))  
        with torch.no_grad():
            output = model(data)
            probabilities = torch.exp(output)  
            probabilities = probabilities.squeeze(0)  

            predicted_class = probabilities.argmax().item()
            confidence_score = probabilities[predicted_class].item()

        return predicted_class, confidence_score
    except Exception as e:
        print(f"Error recognizing gesture: {e}")
        return None, None


image_dir = 'C:\\Users\\truon\OneDrive\\Pictures\\Camera Roll'  
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
i = 0
j = 0
for img_file in image_files:
    img_path = os.path.join(image_dir, img_file)
    image = cv2.imread(img_path)
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            mpDraw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])
                
            landmarks = normalize_row(landmarks)
            if len(landmarks) == 21:
                predicted_class, confidence_score = recognize_gesture(landmarks)
                
                if predicted_class is not None:
                    gesture = 'OK' if predicted_class == 1 else 'None'
                    print(f"Image: {img_file} - Gesture: {gesture} ({confidence_score:.2f})")
                    if gesture == "OK":
                        i += 1
                    else:
                        j += 1
                else:
                    print(f"Image: {img_file} - Insufficient landmarks")
            else:
                print(f"Image: {img_file} - Invalid landmarks count")
    else:
        print(f"Image: {img_file} - No hands detected")

print (i,j)
hands.close()
cv2.destroyAllWindows()
