"""
Image feature extraction for fruit ripeness classification.

This module provides functions for extracting features from fruit images 
that will be used for ripeness classification.
"""

import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage.feature import local_binary_pattern
from utils.image_processing import normalize_image

def extract_features(image):
    """
    Extract features from a fruit image for classification.
    
    Args:
        image (numpy.ndarray): Preprocessed fruit image
        
    Returns:
        numpy.ndarray: Extracted feature vector
    """
    # Extract color features
    color_features = extract_color_features(image)
    
    # Extract texture features
    texture_features = extract_texture_features(image)
    
    # Combine all features
    combined_features = np.concatenate([color_features, texture_features])
    
    return combined_features

def extract_color_features(image):
    """
    Extract color-based features from an image.
    
    Args:
        image (numpy.ndarray): Input image (RGB format)
        
    Returns:
        numpy.ndarray: Color feature vector
    """
    # Ensure the image is in the correct format
    if image.dtype != np.float32 and image.dtype != np.float64:
        image = normalize_image(image.astype(np.float32))
    
    # Convert to uint8 for histogram calculation
    image_uint8 = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    
    # Split into color channels
    r_channel, g_channel, b_channel = cv2.split(image)
    
    # Extract statistics for each channel
    features = []
    
    # Process each channel
    for channel in [r_channel, g_channel, b_channel]:
        # Calculate basic statistics
        mean = np.mean(channel)
        std = np.std(channel)
        median = np.median(channel)
        min_val = np.min(channel)
        max_val = np.max(channel)
        
        # Add to feature vector
        features.extend([mean, std, median, min_val, max_val])
    
    # Convert to HSV color space for additional features
    hsv_image = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv_image)
    
    # Calculate color histograms (RGB)
    hist_bins = 32
    r_hist = cv2.calcHist([image_uint8], [0], None, [hist_bins], [0, 256])
    g_hist = cv2.calcHist([image_uint8], [1], None, [hist_bins], [0, 256])
    b_hist = cv2.calcHist([image_uint8], [2], None, [hist_bins], [0, 256])
    
    # Normalize histograms
    r_hist = cv2.normalize(r_hist, r_hist).flatten()
    g_hist = cv2.normalize(g_hist, g_hist).flatten()
    b_hist = cv2.normalize(b_hist, b_hist).flatten()
    
    # Calculate color histograms (HSV)
    h_hist = cv2.calcHist([hsv_image], [0], None, [hist_bins], [0, 180])
    s_hist = cv2.calcHist([hsv_image], [1], None, [hist_bins], [0, 256])
    v_hist = cv2.calcHist([hsv_image], [2], None, [hist_bins], [0, 256])
    
    # Normalize histograms
    h_hist = cv2.normalize(h_hist, h_hist).flatten()
    s_hist = cv2.normalize(s_hist, s_hist).flatten()
    v_hist = cv2.normalize(v_hist, v_hist).flatten()
    
    # Calculate color ratios
    r_g_ratio = np.mean(r_channel) / (np.mean(g_channel) + 1e-10)  # Avoid division by zero
    r_b_ratio = np.mean(r_channel) / (np.mean(b_channel) + 1e-10)
    g_b_ratio = np.mean(g_channel) / (np.mean(b_channel) + 1e-10)
    
    # Add color ratios to features
    features.extend([r_g_ratio, r_b_ratio, g_b_ratio])
    
    # Add histograms to features (using fewer bins for efficiency)
    reduced_bins = 8
    features.extend(r_hist[::hist_bins//reduced_bins])
    features.extend(g_hist[::hist_bins//reduced_bins])
    features.extend(b_hist[::hist_bins//reduced_bins])
    features.extend(h_hist[::hist_bins//reduced_bins])
    features.extend(s_hist[::hist_bins//reduced_bins])
    
    return np.array(features)

def extract_texture_features(image):
    """
    Extract texture-based features from an image.
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        numpy.ndarray: Texture feature vector
    """
    # Convert to grayscale for texture analysis
    if len(image.shape) == 3:
        if image.dtype == np.float32 or image.dtype == np.float64:
            if image.max() <= 1.0:
                gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Ensure grayscale image is uint8
    if gray.dtype != np.uint8:
        gray = (gray * 255).astype(np.uint8) if gray.max() <= 1.0 else gray.astype(np.uint8)
    
    features = []
    
    # Calculate GLCM (Gray Level Co-occurrence Matrix)
    distances = [1, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    # Reduce gray levels for efficiency and better statistics
    gray_levels = 16
    gray_glcm = (gray // (256 // gray_levels)).astype(np.uint8)
    
    glcm = graycomatrix(gray_glcm, distances=distances, angles=angles, 
                       levels=gray_levels, symmetric=True, normed=True)
    
    # Calculate GLCM properties
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    for prop in props:
        feature = graycoprops(glcm, prop).flatten()
        features.extend(feature)
    
    # Calculate Local Binary Patterns
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    
    # Calculate LBP histogram
    lbp_bins = n_points + 2  # Uniform LBP gives n_points + 2 unique values
    lbp_hist, _ = np.histogram(lbp, bins=lbp_bins, range=(0, lbp_bins), density=True)
    
    features.extend(lbp_hist)
    
    # Add basic texture statistics
    features.append(np.std(gray))  # Standard deviation as a texture measure
    features.append(np.var(gray))  # Variance as a texture measure
    
    return np.array(features)

def generate_feature_importance(model, feature_names=None):
    """
    Generate feature importance scores for the trained model.
    
    Args:
        model: Trained classification model
        feature_names (list): Names of features
        
    Returns:
        dict: Feature importance scores
    """
    # Check if model has feature_importances_ attribute
    if not hasattr(model, 'feature_importances_'):
        return None
    
    importance_scores = model.feature_importances_
    
    # If feature names are provided, create a dictionary
    if feature_names and len(feature_names) == len(importance_scores):
        importance_dict = {name: score for name, score in zip(feature_names, importance_scores)}
        # Sort by importance
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        return sorted_importance
    else:
        # Return raw importance scores
        return importance_scores

def get_importance_map(image, model, grid_size=(16, 16)):
    """
    Generate a feature importance heatmap for the image.
    
    Args:
        image (numpy.ndarray): Input image
        model: Trained classification model
        grid_size (tuple): Size of the grid for importance calculation
        
    Returns:
        numpy.ndarray: Importance map with same dimensions as image
    """
    if not hasattr(model, 'predict_proba'):
        return None
    
    # Get original dimensions
    height, width = image.shape[:2]
    
    # Create grid
    grid_h, grid_w = grid_size
    cell_h, cell_w = height // grid_h, width // grid_w
    
    # Create importance map
    importance_map = np.zeros((height, width), dtype=np.float32)
    
    # Original prediction
    features = extract_features(image)
    original_proba = model.predict_proba([features])[0]
    original_class = np.argmax(original_proba)
    original_confidence = original_proba[original_class]
    
    # Create a copy of the image
    for i in range(grid_h):
        for j in range(grid_w):
            # Create a copy of the image
            modified_image = image.copy()
            
            # Mask out the current grid cell
            y_start, y_end = i * cell_h, (i + 1) * cell_h
            x_start, x_end = j * cell_w, (j + 1) * cell_w
            
            # Apply mask (set to average color or black)
            if len(image.shape) == 3:
                channel_means = np.mean(image, axis=(0, 1))
                modified_image[y_start:y_end, x_start:x_end] = channel_means
            else:
                modified_image[y_start:y_end, x_start:x_end] = 0
            
            # Get prediction for modified image
            mod_features = extract_features(modified_image)
            mod_proba = model.predict_proba([mod_features])[0]
            mod_confidence = mod_proba[original_class]
            
            # Calculate importance (decrease in confidence)
            importance = original_confidence - mod_confidence
            
            # Assign importance to the grid cell
            importance_map[y_start:y_end, x_start:x_end] = importance
    
    # Normalize importance map
    if np.max(importance_map) > 0:
        importance_map = importance_map / np.max(importance_map)
    
    return importance_map
