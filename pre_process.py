import numpy as np
import pandas as pd

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

    normalized_df = normalize_data(input_file)

    sheared_df = shear_data(normalized_df, shear_factor=0.1)
    flipped_df = flip_data(normalized_df)
    translated_df = translate_data(normalized_df, x_shift=0.05, y_shift=0.05, z_shift=0.01)

    combined_df = pd.concat([normalized_df, sheared_df, flipped_df, translated_df], ignore_index=True)

    return combined_df
 
def center_data(df, left_hip_idx=23, right_hip_idx=24):
    centered_df = df.copy()

    left_hip_x = df.iloc[:, 3*left_hip_idx].values
    left_hip_y = df.iloc[:, 3*left_hip_idx + 1].values
    left_hip_z = df.iloc[:, 3*left_hip_idx + 2].values
    
    right_hip_x = df.iloc[:, 3*right_hip_idx].values
    right_hip_y = df.iloc[:, 3*right_hip_idx + 1].values
    right_hip_z = df.iloc[:, 3*right_hip_idx + 2].values

    mid_x = (left_hip_x + right_hip_x) / 2
    mid_y = (left_hip_y + right_hip_y) / 2
    mid_z = (left_hip_z + right_hip_z) / 2

    for i in range(0, df.shape[1], 3):
        centered_df.iloc[:, i] = df.iloc[:, i] - mid_x  
        centered_df.iloc[:, i + 1] = df.iloc[:, i + 1] - mid_y  
        centered_df.iloc[:, i + 2] = df.iloc[:, i + 2] - mid_z  
    
    return centered_df