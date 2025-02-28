"""
Image processing utilities for the Fruit Ripeness Classification System.

This module provides functions for preprocessing, normalizing, and segmenting
fruit images to prepare them for classification.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance


def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess an image for classification.
    
    Args:
        image (PIL.Image or numpy.ndarray): Input image
        target_size (tuple): Target size for resizing (width, height)
        
    Returns:
        numpy.ndarray: Preprocessed image
    """
    # Convert to PIL Image if numpy array
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Resize the image
    image = image.resize(target_size, Image.LANCZOS)
    
    # Convert to numpy array in RGB format
    img_array = np.array(image)
    
    # Ensure image has 3 channels (RGB)
    if len(img_array.shape) == 2:  # Grayscale
        img_array = np.stack((img_array,) * 3, axis=-1)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:, :, :3]
    
    # Normalize pixel values to range [0, 1]
    img_array = normalize_image(img_array)
    
    return img_array


def normalize_image(image):
    """
    Normalize image pixel values to the range [0, 1].
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        numpy.ndarray: Normalized image
    """
    # Convert to float32 for better precision
    image = image.astype(np.float32)
    
    # Normalize to [0, 1]
    if image.max() > 1.0:
        image = image / 255.0
        
    return image


def segment_fruit(image):
    """
    Segment the fruit from the background.
    
    Args:
        image (numpy.ndarray): Input image (BGR format)
        
    Returns:
        numpy.ndarray: Segmented fruit image and the mask
    """
    # Convert to HSV color space for better segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create initial mask with a range that typically catches fruits
    # We use a broader initial range to catch most fruits
    lower_hsv = np.array([0, 30, 30])
    upper_hsv = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # If no contours found, return the original image
        return image, np.ones(image.shape[:2], dtype=np.uint8) * 255
    
    # Find the largest contour (assuming it's the fruit)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Create a refined mask with just the largest contour
    refined_mask = np.zeros_like(mask)
    cv2.drawContours(refined_mask, [largest_contour], 0, 255, -1)
    
    # Apply the mask to the original image
    segmented = cv2.bitwise_and(image, image, mask=refined_mask)
    
    return segmented, refined_mask


def apply_image_transforms(image):
    """
    Apply various transforms to standardize image appearance.
    
    Args:
        image (PIL.Image): Input image
        
    Returns:
        PIL.Image: Transformed image
    """
    # Enhance contrast slightly
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.2)
    
    # Enhance brightness slightly to standardize lighting
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.1)
    
    # Enhance color saturation slightly
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(1.1)
    
    # Apply slight sharpening for better feature detection
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.2)
    
    return image


def extract_image_for_display(image_path, max_size=(400, 400)):
    """
    Load and resize an image for display in the UI.
    
    Args:
        image_path (str): Path to the image file
        max_size (tuple): Maximum dimensions (width, height)
        
    Returns:
        PIL.Image: Resized image for display
    """
    image = Image.open(image_path)
    
    # Calculate scaling factor to fit within max_size while preserving aspect ratio
    width, height = image.size
    scale = min(max_size[0] / width, max_size[1] / height)
    
    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize the image
    display_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    return display_image
