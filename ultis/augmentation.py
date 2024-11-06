import cv2
import numpy as np
import random

def blur_image(image, blur_limit=3):
    ksize = random.choice(range(1, blur_limit, 2))
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def add_noise(image, var=30):
    noise = np.random.normal(0, var, image.shape).astype(np.uint8)
    return cv2.add(image, noise)

def elastic_transform(image, alpha=1, sigma=50, alpha_affine=50):
    random_state = np.random.RandomState(None)
    shape = image.shape
    dx = random_state.randn(*shape[:2]) * sigma
    dy = random_state.randn(*shape[:2]) * sigma
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

def adjust_brightness_contrast(image, brightness=0.3, contrast=0.3):
    alpha = 1.0 + random.uniform(-contrast, contrast)
    beta = random.uniform(-brightness * 255, brightness * 255)
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def resize_far(image, scale_factor=0.5):
    original_h, original_w = image.shape[:2]

    new_h, new_w = int(original_h * scale_factor), int(original_w * scale_factor)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    far_away_image = np.zeros_like(image)

    top_pad = (original_h - new_h) // 2
    left_pad = (original_w - new_w) // 2
    
    far_away_image[top_pad:top_pad + new_h, left_pad:left_pad + new_w] = resized_image
    
    return far_away_image

def augment_image(image):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    if random.random() > 0.5:
        image = cv2.flip(image, 1)

    if random.random() > 0.5:
        image = resize_far(image, scale_factor=random.uniform(0.2, 0.6))

    angle = random.randint(-15, 15)
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h))

    value = random.randint(-30, 30)
    hsv[:, :, 2] = cv2.add(hsv[:, :, 2], value)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    crop_percent = random.uniform(0.8, 1.0)
    new_h, new_w = int(h * crop_percent), int(w * crop_percent)
    top, left = random.randint(0, h - new_h), random.randint(0, w - new_w)
    image = image[top:top + new_h, left:left + new_w]
    image = cv2.resize(image, (w, h))

    image = blur_image(image)
    image = add_noise(image)
    image = elastic_transform(image)
    image = adjust_brightness_contrast(image)

    return image

def augment_images(images, num_augmented=5):
    augmented_images = []
    for img in images:
        augmented_images.append(img)
        for _ in range(num_augmented):
            aug_img = augment_image(img)
            augmented_images.append(aug_img)
    return augmented_images
