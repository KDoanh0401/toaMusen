import torch 
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Dropout
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def center_data(landmarks):
    data = np.array(landmarks).reshape(33, 4).T
    for axis in range(3):  
            data[axis, :] = data[axis, :] - (data[axis, 23] + data[axis, 24]) / 2

    centered_row = data.T
    return centered_row

def normalize_data(landmarks):
    data = np.array(landmarks).reshape(33, 4).T
    for axis in range(3):  
        axis_min = data[axis, :].min()
        axis_max = data[axis, :].max()
        if axis_max - axis_min != 0:
            data[axis, :] = (data[axis, :] - axis_min) / (axis_max - axis_min)
        else:
            data[axis, :] = 0  

    normalized_row = data.T
    return normalized_row 

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
    
def create_edges():
    body_connections = [
    (11, 13), (13, 15), (15, 17), (17, 19), (15, 21),(15, 19),                              
    (12, 14), (14, 16), (16, 18), (18, 20), (20, 16), (16,22),  
    (11, 12)
    ]

    edge_index = torch.tensor(body_connections, dtype=torch.long).t().contiguous() 
    return edge_index.to(device)

model = GCN(num_features=4, num_classes=3).to(device) 
model.load_state_dict(torch.load('model\gcn_pose_gesture_model.pth', map_location=device))
model.eval()

edge_index = create_edges()

def recognize_pose(landmarks, confidence_threshold=0.5):
    if landmarks == None:
        return None, None
    landmarks_df = center_data(landmarks)
    landmarks_df = normalize_data(landmarks_df)
    landmarks_df = torch.tensor(landmarks_df, dtype=torch.float).to(device)
    graph = Data(x=landmarks_df,edge_index=edge_index)   
    with torch.no_grad():
        output = model(graph)
        probabilities = torch.exp(output)
        predicted_class = probabilities.argmax(dim=1).item()
        confidence_score = probabilities.max(dim=1).values.item()
    
    if confidence_score > confidence_threshold:
        match predicted_class:
            case 0: 
                gesture = 'NO'

            case 1:
                gesture = 'OK'
            case 2:
                gesture = 'Unidentified Pose'
        if gesture != 'Unidentified Pose':
            return gesture, confidence_score
        else:
            return gesture, None
    else: 
        return 'Unidentified Pose', None