import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.utils import compute_class_weight
import os
from pre_process import process_data
from torch.nn import BatchNorm1d, Dropout

def load_data(file_path):
    data = pd.read_csv(file_path, header=None)
    return data

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
    
def normalize_data(df):
    def normalize_row(row):

        data = row.values.reshape(3, 21)

        wrist_z = data[2, 0]
        data[2, :] = data[2, :] - wrist_z

        for axis in range(3):
            axis_min = data[axis, :].min()
            axis_max = data[axis, :].max()
            if axis_max - axis_min != 0:
                data[axis, :] = (data[axis, :] - axis_min) / (axis_max - axis_min)
            else:
                data[axis, :] = 0
        
        normalized_row = data.flatten()
        return normalized_row
    
    normalized_array = df.apply(normalize_row, axis=1, result_type='expand').to_numpy()
    
    normalized_df = pd.DataFrame(normalized_array, columns=df.columns)
    
    return normalized_df

def prepare_data(data_frame, labels):
    graph_data = []
    edge_index = create_edges()
    
    for index, row in data_frame.iterrows():
        landmarks = torch.tensor(row.values.reshape(21, 3), dtype=torch.float)

        label_holder = torch.tensor([labels[index]], dtype=torch.long)

        graph = Data(x=landmarks, edge_index=edge_index, y=label_holder)
        graph_data.append(graph)
    
    return graph_data

def train_gcn_model(train_data, train_labels, val_data, val_labels, num_epochs=100, batch_size=32, patience=10, retrain=False):

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    model = GCN(num_features=3, num_classes=2)
    model_path = 'gcn_hand_gesture_model.pth'
    if retrain and os.path.exists(model_path):
        print("Loading existing model for retraining...")
        model.load_state_dict(torch.load(model_path))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = F.nll_loss(out, batch.y, weight=class_weights_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss = evaluate_model(model, val_loader, class_weights_tensor, return_loss=True)
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping")
            break
        scheduler.step()

    evaluate_model(model, val_loader, class_weights_tensor)

def evaluate_model(model, loader, class_weights_tensor, return_loss=False):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            out = model(batch)
            loss = F.nll_loss(out, batch.y, weight=class_weights_tensor)
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())

    if return_loss:
        return total_loss / len(loader)

    print(classification_report(all_labels, all_preds, target_names=['None', 'OK']))

ok_file = "test_data\left.csv"
no_file = "test_data\\None_left.csv"
ok_val_file = "test_data\\val_left.csv"
no_val_file = "test_data\\val_None_left.csv"
# none_file = "hand_gesture_data\_None.csv"
# none_val_file = "val_dataset\_None_val.csv"

ok_data = load_data(ok_file)
no_data = load_data(no_file)
# none_data = load_data(none_file)

ok_val_data = load_data(ok_val_file)
no_val_data = load_data(no_val_file)
# none_val_data = load_data(none_val_file)

ok_data = process_data(ok_data)
no_data = process_data(no_data)
# none_data = process_data(none_data)
train_landmark_data = pd.concat([ok_data, no_data], ignore_index=True)

val_landmark_data = pd.concat([ok_val_data, no_val_data], ignore_index=True)
val_landmark_data = normalize_data(val_landmark_data)

train_labels = np.array([1] * len(ok_data) + [0] * len(no_data))
val_labels = np.array([1] * len(ok_val_data) + [0] * len(no_val_data))

train_graph_data = prepare_data(train_landmark_data, train_labels)
val_graph_data = prepare_data(val_landmark_data, val_labels)

train_gcn_model(train_graph_data, train_labels, val_graph_data, val_labels, num_epochs=50, batch_size=128, retrain=False)
