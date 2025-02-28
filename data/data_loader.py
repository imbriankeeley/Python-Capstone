"""
Functions for loading and processing images for the fruit ripeness classifier.

This module provides functions for loading individual images and training datasets.
"""

import os
import numpy as np
import logging
import glob
from PIL import Image
import cv2
from utils.image_processing import preprocess_image, normalize_image

logger = logging.getLogger(__name__)

def load_image(file_path):
    """
    Load an image from the given file path and preprocess it for analysis.
    
    Args:
        file_path (str): Path to the image file
        
    Returns:
        numpy.ndarray: Preprocessed image ready for classification
    """
    try:
        # Load image using PIL
        image = Image.open(file_path)
        
        # Apply preprocessing
        processed_image = preprocess_image(image)
        
        logger.debug(f"Successfully loaded and preprocessed image: {file_path}")
        return processed_image
    
    except Exception as e:
        logger.error(f"Error loading image {file_path}: {str(e)}")
        raise

def load_training_dataset(dataset_dir, categories=None):
    """
    Load the training dataset from the specified directory.
    
    Expected directory structure:
    dataset_dir/
        unripe/
            image1.jpg
            image2.jpg
            ...
        ripe/
            image1.jpg
            ...
        overripe/
            image1.jpg
            ...
        spoiled/
            image1.jpg
            ...
    
    Args:
        dataset_dir (str): Directory containing training images
        categories (list): List of category names. If None, uses default categories.
        
    Returns:
        tuple: (images, labels) arrays for training
    """
    if categories is None:
        categories = ["unripe", "ripe", "overripe", "spoiled"]
    
    images = []
    labels = []
    
    logger.info(f"Loading training dataset from {dataset_dir}")
    
    # Check if the directory exists
    if not os.path.exists(dataset_dir):
        logger.error(f"Dataset directory not found: {dataset_dir}")
        return np.array([]), np.array([])
    
    # Load images from each category
    for category_index, category in enumerate(categories):
        category_dir = os.path.join(dataset_dir, category)
        
        # Check if category directory exists
        if not os.path.exists(category_dir):
            logger.warning(f"Category directory not found: {category_dir}")
            continue
        
        # Get all image files in the category directory
        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            image_files.extend(glob.glob(os.path.join(category_dir, ext)))
        
        logger.info(f"Found {len(image_files)} images in category '{category}'")
        
        # Load and preprocess each image
        for image_file in image_files:
            try:
                # Load and preprocess the image
                processed_image = load_image(image_file)
                
                # Add image and label to the dataset
                images.append(processed_image)
                labels.append(category_index)
                
            except Exception as e:
                logger.warning(f"Error processing image {image_file}: {str(e)}")
                continue
    
    # Convert lists to numpy arrays
    if images:
        images_array = np.array(images)
        labels_array = np.array(labels)
        
        logger.info(f"Successfully loaded {len(images)} images from {len(categories)} categories")
        
        return images_array, labels_array
    else:
        logger.warning("No images were loaded from the dataset directory")
        return np.array([]), np.array([])

def save_processed_image(image, file_path):
    """
    Save a processed image to disk.
    
    Args:
        image (numpy.ndarray): Processed image
        file_path (str): Target file path for saving
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Convert to PIL Image and save
        if image.dtype == np.float32 or image.dtype == np.float64:
            # Convert from [0,1] to [0,255] if necessary
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
        
        # Convert to RGB if it's in BGR format (from OpenCV)
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Check if it might be in BGR format (OpenCV default)
            # This is a heuristic and might not be perfect
            if cv2.cvtColor(image, cv2.COLOR_BGR2RGB).std() < image.std():
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Save the image
        Image.fromarray(image).save(file_path)
        logger.debug(f"Image saved to {file_path}")
        
    except Exception as e:
        logger.error(f"Error saving image to {file_path}: {str(e)}")
        raise

def get_sample_images(dataset_dir, categories=None, num_samples=1):
    """
    Get sample images from each category in the dataset.
    
    Args:
        dataset_dir (str): Directory containing training images
        categories (list): List of category names. If None, uses default categories.
        num_samples (int): Number of samples to get from each category
        
    Returns:
        dict: Dictionary mapping categories to lists of sample images
    """
    if categories is None:
        categories = ["unripe", "ripe", "overripe", "spoiled"]
    
    samples = {}
    
    for category in categories:
        category_dir = os.path.join(dataset_dir, category)
        
        # Check if category directory exists
        if not os.path.exists(category_dir):
            continue
        
        # Get all image files in the category directory
        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            image_files.extend(glob.glob(os.path.join(category_dir, ext)))
        
        # Get random samples
        if image_files:
            # Get at most num_samples images or all available if fewer
            sample_count = min(num_samples, len(image_files))
            indices = np.random.choice(len(image_files), sample_count, replace=False)
            sample_files = [image_files[i] for i in indices]
            
            # Load the samples
            samples[category] = []
            for sample_file in sample_files:
                try:
                    image = Image.open(sample_file)
                    samples[category].append(image)
                except Exception:
                    continue
    
    return samples
