import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def normalize_data(df):
    normalized_array = df.copy()
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
        
        return data.flatten()
    
    normalized_array = normalized_array.apply(normalize_row, axis=1, result_type='expand').to_numpy()
    normalized_df = pd.DataFrame(normalized_array, columns=df.columns)
    
    return normalized_df
def normalize_hand_data(hand_data):
    hand_data = hand_data.reshape(3, 21)  
    for axis in range(3):  
        axis_min = hand_data[axis, :].min()
        axis_max = hand_data[axis, :].max()
        if axis_max - axis_min != 0:
            hand_data[axis, :] = (hand_data[axis, :] - axis_min) / (axis_max - axis_min)
        else:
            hand_data[axis, :] = 0  

    return hand_data.flatten()

def normalize_both_hands(data):
    normalized_data = []

    for _, row in data.iterrows():
        left_hand = row[:63].values  
        right_hand = row[63:].values  

        left_hand_normalized = normalize_hand_data(left_hand)
        right_hand_normalized = normalize_hand_data(right_hand)

        normalized_row = np.concatenate([left_hand_normalized, right_hand_normalized])
        normalized_data.append(normalized_row)

    normalized_df = pd.DataFrame(normalized_data, columns=data.columns)
    return normalized_df

def shear_data(df, shear_factor=0.1):
    sheared_df = df.copy()
    shear_matrix = np.array([[1, shear_factor, 0], [0, 1, 0], [0, 0, 1]])  
    for i in range(0, df.shape[1], 3):  
        x_vals = df.iloc[:, i].values
        y_vals = df.iloc[:, i + 1].values
        z_vals = df.iloc[:, i + 2].values
        coordinates = np.stack([x_vals, y_vals, z_vals], axis=1)
        sheared_coords = np.dot(coordinates, shear_matrix.T)
        sheared_df.iloc[:, i] = sheared_coords[:, 0]  
        sheared_df.iloc[:, i + 1] = sheared_coords[:, 1]  
        sheared_df.iloc[:, i + 2] = sheared_coords[:, 2]  
    return sheared_df

def translate_data(df, x_shift=0.05, y_shift=0.05, z_shift=0.0):
    translated_df = df.copy()
    for i in range(0, df.shape[1], 3):
        translated_df.iloc[:, i] = df.iloc[:, i] + x_shift  
        translated_df.iloc[:, i + 1] = df.iloc[:, i + 1] + y_shift  
        translated_df.iloc[:, i + 2] = df.iloc[:, i + 2] + z_shift 
    return translated_df

def visualize_hand_landmarks(dfs, labels, colors, hand_connections, row_indices=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if row_indices is None:
        row_indices = range(len(dfs[0]))

    for df, label, color in zip(dfs, labels, colors):
        for idx in row_indices:
            row = df.iloc[idx]
            xs = row[::3].values
            ys = row[1::3].values
            zs = row[2::3].values

            ax.scatter(xs, ys, zs, c=color, marker='o', label=label if idx == row_indices[0] else "")

            for connection in hand_connections:
                start_idx, end_idx = connection
                ax.plot(
                    [xs[start_idx], xs[end_idx]],
                    [ys[start_idx], ys[end_idx]],
                    [zs[start_idx], zs[end_idx]],
                    color=color,
                    linewidth=2,
                    label=label if idx == row_indices[0] and connection == hand_connections[0] else ""
                )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])

    handles, labels_ = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_, handles))
    ax.legend(by_label.values(), by_label.keys())

    plt.title('Comparison of Hand Landmarks')
    plt.show()

def visualize_both_hands(data, hand_connections, row_index=0):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    row = data.iloc[row_index].values
    
    left_hand_xs, left_hand_ys, left_hand_zs = row[0:63:3], row[1:63:3], row[2:63:3]
    right_hand_xs, right_hand_ys, right_hand_zs = row[63::3], row[64::3], row[65::3]

    ax.scatter(left_hand_xs, left_hand_ys, left_hand_zs, c='blue', marker='o', label='Left Hand')
    for connection in hand_connections:
        start_idx, end_idx = connection
        ax.plot([left_hand_xs[start_idx], left_hand_xs[end_idx]],
                [left_hand_ys[start_idx], left_hand_ys[end_idx]],
                [left_hand_zs[start_idx], left_hand_zs[end_idx]], color='blue', linewidth=2)

    ax.scatter(right_hand_xs, right_hand_ys, right_hand_zs, c='red', marker='o', label='Right Hand')
    for connection in hand_connections:
        start_idx, end_idx = connection
        ax.plot([right_hand_xs[start_idx], right_hand_xs[end_idx]],
                [right_hand_ys[start_idx], right_hand_ys[end_idx]],
                [right_hand_zs[start_idx], right_hand_zs[end_idx]], color='red', linewidth=2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])

    ax.legend()
    plt.title(f'Visualization of Both Hands (Row {row_index})')
    plt.show()


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

hi_df = pd.read_csv('hand_gesture_data\_NO.csv.')
ok_df = pd.read_csv('hand_gesture_data\OK.csv')

hi_normalized = normalize_both_hands(hi_df)
ok_normalized = normalize_both_hands(ok_df)

hi_sheared = shear_data(hi_normalized, shear_factor=0.1)
ok_sheared = shear_data(ok_normalized, shear_factor=0.1)

hi_translated = translate_data(hi_normalized)
dfs_to_plot = [hi_df]
labels = ['raw']
colors = ['blue']
row_indices = [25] 
# visualize_hand_landmarks(dfs_to_plot, labels, colors, hand_connections, row_indices=row_indices)
visualize_both_hands(ok_df, hand_connections, row_index=600)