import torch
import cv2
import mediapipe as mp
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Dropout

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize_row(row):
    data = np.array(row).reshape(3, 42)
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

        self.dropout = Dropout(0.5)  

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)  

        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = self.dropout(x) 
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch) 
        return F.log_softmax(x, dim=1)  
    
model = GCN(num_features=3, num_classes=2).to(device)
model.load_state_dict(torch.load('gcn_hand_gesture_model.pth',map_location=device))
model.eval()

def create_edges():
    hand_connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20)
    ]
    edge_index = torch.tensor(hand_connections, dtype=torch.long).t().contiguous()
    return edge_index.to(device)

edge_index = create_edges()

def recognize_hand_gesture(landmarks, confidence_threshold=0.5):
    landmarks_df = normalize_row(landmarks)
    landmarks_df = torch.tensor(landmarks_df, dtype=torch.float).to(device)
            
    graph = Data(x=landmarks_df, edge_index=edge_index)

    with torch.no_grad():
        output = model(graph)
        probabilities = torch.exp(output)
        predicted_class = probabilities.argmax(dim=1).item()
        confidence_score = probabilities.max(dim=1).values.item()

    if confidence_score > confidence_threshold:
        gesture = 'NO' if predicted_class == 1 else 'None'

    return gesture, confidence_score

