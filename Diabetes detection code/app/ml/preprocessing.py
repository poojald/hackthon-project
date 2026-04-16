import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess retinal image for model input
    
    Args:
        image_path: Path to the image file
        target_size: Target size for resizing (height, width)
    
    Returns:
        Preprocessed image array ready for model input
    """
    # Read image
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Apply preprocessing pipeline
    img = remove_artifacts(img)
    img = enhance_contrast(img)
    img = normalize_illumination(img)
    
    # Resize to target size
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)
    
    # Normalize pixel values to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img


def remove_artifacts(image):
    """
    Remove artifacts and noise from retinal images
    
    Args:
        image: Input image (RGB)
    
    Returns:
        Cleaned image
    """
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Apply bilateral filter to preserve edges while smoothing
    cleaned = cv2.bilateralFilter(blurred, 9, 75, 75)
    
    return cleaned


def enhance_contrast(image):
    """
    Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    
    Args:
        image: Input image (RGB)
    
    Returns:
        Contrast-enhanced image
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Split channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    # Merge channels
    enhanced_lab = cv2.merge([l_enhanced, a, b])
    
    # Convert back to RGB
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    
    return enhanced


def normalize_illumination(image):
    """
    Normalize illumination across the image
    
    Args:
        image: Input image (RGB)
    
    Returns:
        Illumination-normalized image
    """
    # Convert to float
    img_float = image.astype(np.float32)
    
    # Apply morphological opening to estimate background
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    background = cv2.morphologyEx(img_float, cv2.MORPH_OPEN, kernel)
    
    # Subtract background
    normalized = cv2.subtract(img_float, background)
    
    # Normalize to [0, 255]
    normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)
    normalized = normalized.astype(np.uint8)
    
    return normalized


def crop_to_roi(image):
    """
    Crop image to region of interest (remove black borders)
    
    Args:
        image: Input image (RGB)
    
    Returns:
        Cropped image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Threshold to find non-black regions
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get bounding box of largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Crop image
        cropped = image[y:y+h, x:x+w]
        return cropped
    
    return image


def augment_image(image):
    """
    Apply data augmentation to image
    
    Args:
        image: Input image array
    
    Returns:
        Augmented image
    """
    # Random rotation
    if np.random.random() > 0.5:
        angle = np.random.uniform(-15, 15)
        image = rotate_image(image, angle)
    
    # Random horizontal flip
    if np.random.random() > 0.5:
        image = cv2.flip(image, 1)
    
    # Random vertical flip
    if np.random.random() > 0.5:
        image = cv2.flip(image, 0)
    
    # Random brightness adjustment
    if np.random.random() > 0.5:
        factor = np.random.uniform(0.8, 1.2)
        image = adjust_brightness(image, factor)
    
    # Random contrast adjustment
    if np.random.random() > 0.5:
        factor = np.random.uniform(0.8, 1.2)
        image = adjust_contrast(image, factor)
    
    # Random zoom
    if np.random.random() > 0.5:
        zoom_factor = np.random.uniform(0.9, 1.1)
        image = zoom_image(image, zoom_factor)
    
    return image


def rotate_image(image, angle):
    """Rotate image by given angle"""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT)
    
    return rotated


def adjust_brightness(image, factor):
    """Adjust image brightness"""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    hsv = hsv.astype(np.uint8)
    
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def adjust_contrast(image, factor):
    """Adjust image contrast"""
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    adjusted = (image - mean) * factor + mean
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    
    return adjusted


def zoom_image(image, zoom_factor):
    """Zoom in/out on image"""
    height, width = image.shape[:2]
    
    # Calculate new dimensions
    new_height = int(height * zoom_factor)
    new_width = int(width * zoom_factor)
    
    # Resize
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # Crop or pad to original size
    if zoom_factor > 1:
        # Crop center
        start_y = (new_height - height) // 2
        start_x = (new_width - width) // 2
        result = resized[start_y:start_y+height, start_x:start_x+width]
    else:
        # Pad
        result = np.zeros((height, width, 3), dtype=np.uint8)
        start_y = (height - new_height) // 2
        start_x = (width - new_width) // 2
        result[start_y:start_y+new_height, start_x:start_x+new_width] = resized
    
    return result


def create_augmentation_pipeline():
    """
    Create TensorFlow data augmentation pipeline
    
    Returns:
        Sequential model for data augmentation
    """
    from tensorflow.keras import layers
    
    augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2),
    ], name='data_augmentation')
    
    return augmentation


def batch_preprocess_images(image_paths, target_size=(224, 224)):
    """
    Preprocess multiple images in batch
    
    Args:
        image_paths: List of image file paths
        target_size: Target size for resizing
    
    Returns:
        Batch of preprocessed images
    """
    images = []
    
    for path in image_paths:
        try:
            img = preprocess_image(path, target_size)
            images.append(img[0])  # Remove batch dimension
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
            continue
    
    if not images:
        return None
    
    # Stack images into batch
    batch = np.stack(images, axis=0)
    
    return batch
