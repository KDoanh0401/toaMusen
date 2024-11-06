import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

body_connections = [
        (0,1), (1,2), (2,3), (3,7),
        (0,4), (4,5), (5,6), (6,8),
        (9,10),
        (11,13), (13,15), (15,21), (15,19), (15,17), (17,19),
        (12,14), (14,16), (16,22), (16,20), (16,18), (18,20),
        (12,24), (24,26), (26,28), (28,30), (28,32), (30,32),
        (11,23), (23,25), (25,27), (27,29), (27,31), (29,31),
        (11,12),
        (23,24)
    ]

def center_data(df):
    centered_df = df.copy()

    def center_row(row):
        
        data = row.values.reshape(33, 4).T
        for axis in range(3):  
                data[axis, :] = data[axis, :] - (data[axis, 23] + data[axis, 24]) / 2

        centered_row = data.T.flatten()
        return centered_row

    centered_df = centered_df.apply(center_row, axis=1, result_type='expand')

    return centered_df

def normalize_data(df):
    normalized_df = df.copy()
    
    def normalize_row(row):

        data = row.values.reshape(33, 4).T
        for axis in range(3):  
            axis_min = data[axis, :].min()
            axis_max = data[axis, :].max()
            if axis_max - axis_min != 0:
                data[axis, :] = (data[axis, :] - axis_min) / (axis_max - axis_min)
            else:
                data[axis, :] = 0  

        normalized_row = data.T.flatten()
        return normalized_row

    normalized_array = normalized_df.apply(normalize_row, axis=1, result_type='expand').to_numpy()

    normalized_df = pd.DataFrame(normalized_array, columns=df.columns)
    
    return normalized_df

def visualize_hand_landmarks(dfs, labels, colors, body_connections, frame_indices=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if frame_indices is None:
        frame_indices = range(len(dfs[0]))

    for df, label, color in zip(dfs, labels, colors):
        for frame_idx in frame_indices:
            frame_data = df.iloc[frame_idx]

            # Extract x, y, z coordinates from the frame
            xs = frame_data[::4].values
            ys = frame_data[1::4].values
            zs = frame_data[2::4].values
            var = len(xs)
            for i in range(var):
                print(i,xs[i],ys[i],zs[i])
            # Scatter plot for landmarks
            ax.scatter(xs, ys, zs, c=color, marker='o', s=40, label=label if frame_idx == frame_indices[0] else "")

            # Label each landmark with its index
            for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
                ax.text(x, y, z, f'{i}', color='black', fontsize=8, ha='center', va='center')

            # Draw connections between landmarks
            for connection in body_connections:
                start_idx, end_idx = connection
                ax.plot(
                    [xs[start_idx], xs[end_idx]],
                    [ys[start_idx], ys[end_idx]],
                    [zs[start_idx], zs[end_idx]],
                    color=color,
                    linewidth=2
                )

    # Set axis labels and limits
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])

    # Create a legend for gesture labels
    handles, labels_ = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_, handles))
    ax.legend(by_label.values(), by_label.keys())

    plt.title('Hand Landmarks Visualization with Labels')
    plt.show()

def shear_data(df, shear_factor=0.1):
    sheared_df = df.copy()

    if df.shape[1] % 3 != 0:
        raise ValueError("Number of columns must be divisible by 3 (x, y, z coordinates).")
    
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

def flip_data(df):

    flipped_df = df.copy()

    for i in range(0, df.shape[1], 3):
        flipped_df.iloc[:, i] = 1 - df.iloc[:, i]

    return flipped_df

def translate_data(df, x_shift=0.05, y_shift=0.05, z_shift=0.0):
    translated_df = df.copy()

    for i in range(0, df.shape[1], 3):

        translated_df.iloc[:, i] = df.iloc[:, i] + x_shift

        translated_df.iloc[:, i + 1] = df.iloc[:, i + 1] + y_shift

        translated_df.iloc[:, i + 2] = df.iloc[:, i + 2] + z_shift

    return translated_df

def rotate_data(df, angle_degrees=15, axis='z', reference_point_idx=0):

    rotated_df = df.copy()

    angle_radians = np.radians(angle_degrees)
    
    if axis == 'x':
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, np.cos(angle_radians), -np.sin(angle_radians)],
                                    [0, np.sin(angle_radians), np.cos(angle_radians)]])
    elif axis == 'y':
        rotation_matrix = np.array([[np.cos(angle_radians), 0, np.sin(angle_radians)],
                                    [0, 1, 0],
                                    [-np.sin(angle_radians), 0, np.cos(angle_radians)]])
    elif axis == 'z':
        rotation_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians), 0],
                                    [np.sin(angle_radians), np.cos(angle_radians), 0],
                                    [0, 0, 1]])
    else:
        raise ValueError("Invalid axis. Choose from 'x', 'y', or 'z'.")
    
    ref_x = df.iloc[:, 3*reference_point_idx].values
    ref_y = df.iloc[:, 3*reference_point_idx + 1].values
    ref_z = df.iloc[:, 3*reference_point_idx + 2].values
    
    for i in range(0, df.shape[1], 3):
        rotated_df.iloc[:, i] = df.iloc[:, i] - ref_x
        rotated_df.iloc[:, i + 1] = df.iloc[:, i + 1] - ref_y
        rotated_df.iloc[:, i + 2] = df.iloc[:, i + 2] - ref_z

    for i in range(0, df.shape[1], 3):
        x_vals = rotated_df.iloc[:, i].values
        y_vals = rotated_df.iloc[:, i + 1].values
        z_vals = rotated_df.iloc[:, i + 2].values
        coordinates = np.stack([x_vals, y_vals, z_vals], axis=1) 
        
        rotated_coords = np.dot(coordinates, rotation_matrix.T)
        
        rotated_df.iloc[:, i] = rotated_coords[:, 0] 
        rotated_df.iloc[:, i + 1] = rotated_coords[:, 1]  
        rotated_df.iloc[:, i + 2] = rotated_coords[:, 2]  
    
    for i in range(0, df.shape[1], 3):
        rotated_df.iloc[:, i] = rotated_df.iloc[:, i] + ref_x
        rotated_df.iloc[:, i + 1] = rotated_df.iloc[:, i + 1] + ref_y
        rotated_df.iloc[:, i + 2] = rotated_df.iloc[:, i + 2] + ref_z
    
    return rotated_df

def process_data(input_file):
    center_df = center_data(input_file)
    normalized_df = normalize_data(center_df)

    sheared_df = shear_data(normalized_df, shear_factor=0.1)
    flipped_df = flip_data(normalized_df)
    rotated_df  = rotate_data(normalized_df)
    translated_df = translate_data(normalized_df, x_shift=0.05, y_shift=0.05, z_shift=0.01)

    combined_df = pd.concat([normalized_df, sheared_df, rotated_df, translated_df], ignore_index=True)

    return combined_df

# df1 = pd.read_csv('pose_data\poseOK.csv')
# df1 = df1.drop(columns=df1.columns[0])  
# centered_df = center_data(df1)
# print(centered_df)
# with open('first_row.txt', 'w') as f:
#     f.write(centered_df.iloc[0].to_string())

# normalized_df = normalize_data(centered_df)
# labels = ['centered_df']
# dfs = [centered_df]
# # Colors for each gesture
# colors = ['red']
# visualize_hand_landmarks(dfs, labels, colors, body_connections, [0])