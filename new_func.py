# import cv2
# import numpy as np
# import random

# def scale_image(image, scale_range=(0.2, 0.2)):
#     scale = random.uniform(*scale_range)
#     h, w = image.shape[:2]
#     new_h, new_w = int(h * scale), int(w * scale)
#     scaled_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

#     if scale > 1.0:
#         start_h, start_w = (new_h - h) // 2, (new_w - w) // 2
#         return scaled_image[start_h:start_h + h, start_w:start_w + w]
#     else:
#         pad_h, pad_w = (h - new_h) // 2, (w - new_w) // 2
#         return cv2.copyMakeBorder(scaled_image, pad_h, h - new_h - pad_h, pad_w, w - new_w - pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))

# def apply_gaussian_blur(image, blur_limit=(1, 3)):
#     ksize = random.choice(range(*blur_limit))
#     return cv2.GaussianBlur(image, (ksize * 2 + 1, ksize * 2 + 1), 0)

# def adjust_brightness_contrast(image, brightness_range=(-30, 30), contrast_range=(0.8, 1.2)):
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     brightness = random.randint(*brightness_range)
#     contrast = random.uniform(*contrast_range)
#     hsv[:, :, 2] = np.clip(hsv[:, :, 2] * contrast + brightness, 0, 255).astype(np.uint8)
#     return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# def random_crop_pad(image, crop_range=(0.9, 1.0)):
#     crop_scale = random.uniform(*crop_range)
#     h, w = image.shape[:2]
#     new_h, new_w = int(h * crop_scale), int(w * crop_scale)
#     top, left = random.randint(0, h - new_h), random.randint(0, w - new_w)
    
#     cropped_image = image[top:top + new_h, left:left + new_w]
#     return cv2.resize(cropped_image, (w, h))

# def augment_face_image(image):
#     image = scale_image(image)
#     image = apply_gaussian_blur(image)
#     return image

# input_image = cv2.imread('D:\ToaDA\Recognition\Doanh\Doanh (1).jpg')  
# augmented_image = augment_face_image(input_image)

# # cv2.imshow('Original Image', input_image)
# cv2.imshow('Augmented Image', augmented_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import os

video_path = 'C:\\Users\\truon\\OneDrive\\Pictures\\Camera Roll\\abc.mp4'  
output_folder = 'Doanh'        
frame_interval = 15                 
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_count = 0 
saved_count = 0 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break 

    if frame_count % frame_interval == 0:
        frame_filename = os.path.join(output_folder, f'frame_{frame_count}.jpg')
        cv2.imwrite(frame_filename, frame)
        saved_count += 1
        print(f'Saved: {frame_filename}')
    
    frame_count += 1

cap.release()
print(f'Total frames saved: {saved_count}')