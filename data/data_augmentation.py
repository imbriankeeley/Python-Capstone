"""
Functions for data augmentation to expand the training dataset.

File path: /data/data_augmentation.py
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import random
import logging
import cv2

logger = logging.getLogger(__name__)

def augment_image(image, augmentation_level=0.5):
    """
    Apply random augmentation to an image.
    
    Args:
        image (PIL.Image): Original image
        augmentation_level (float): Strength of augmentation (0-1)
        
    Returns:
        PIL.Image: Augmented image
    """
    # Input validation
    if not isinstance(image, Image.Image):
        raise TypeError("Input must be a PIL Image")
    
    # Make a copy to avoid modifying the original
    img = image.copy()
    
    # Apply random rotation (up to 20 degrees)
    if random.random() < augmentation_level:
        rotation_angle = random.uniform(-20, 20)
        img = img.rotate(rotation_angle, resample=Image.BICUBIC, expand=False)
    
    # Random horizontal flip
    if random.random() < augmentation_level * 0.7:
        img = ImageOps.mirror(img)
    
    # Random brightness adjustment
    if random.random() < augmentation_level:
        brightness_factor = random.uniform(0.85, 1.15)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness_factor)
    
    # Random contrast adjustment
    if random.random() < augmentation_level:
        contrast_factor = random.uniform(0.85, 1.15)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast_factor)
    
    # Random color (saturation) adjustment
    if random.random() < augmentation_level:
        color_factor = random.uniform(0.85, 1.15)
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(color_factor)
    
    # Random sharpness adjustment
    if random.random() < augmentation_level:
        sharpness_factor = random.uniform(0.85, 1.15)
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(sharpness_factor)
    
    # Random crop and resize (slight zoom effect)
    if random.random() < augmentation_level * 0.5:
        width, height = img.size
        crop_percentage = random.uniform(0.85, 0.95)
        crop_width = int(width * crop_percentage)
        crop_height = int(height * crop_percentage)
        
        # Calculate random crop position
        left = random.randint(0, width - crop_width)
        top = random.randint(0, height - crop_height)
        right = left + crop_width
        bottom = top + crop_height
        
        # Crop and resize back to original size
        img = img.crop((left, top, right, bottom))
        img = img.resize((width, height), Image.LANCZOS)
    
    # Random slight perspective transformation
    if random.random() < augmentation_level * 0.3:
        width, height = img.size
        
        # Define slight perspective distortion
        distortion = 0.05
        
        # Define the 8 corner points for perspective transform
        src_points = [(0, 0), (width, 0), (width, height), (0, height)]
        
        # Slightly move each corner for the destination points
        dst_points = [
            (random.uniform(0, width * distortion), random.uniform(0, height * distortion)),
            (random.uniform(width * (1 - distortion), width), random.uniform(0, height * distortion)),
            (random.uniform(width * (1 - distortion), width), random.uniform(height * (1 - distortion), height)),
            (random.uniform(0, width * distortion), random.uniform(height * (1 - distortion), height))
        ]
        
        # Calculate the perspective transform matrix
        coeffs = _find_coeffs(dst_points, src_points)
        
        # Apply the transformation
        img = img.transform((width, height), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
    
    return img

def _find_coeffs(pa, pb):
    """
    Helper function to find coefficients for perspective transformation.
    
    Args:
        pa: Points in the output plane
        pb: Corresponding points in the input plane
        
    Returns:
        list: Coefficients for the transformation
    """
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=np.float32)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)

def create_ripeness_gradient(image, ripeness_level):
    """
    Create a simulated image for different ripeness levels.
    Used to expand binary classification to four categories.
    
    Args:
        image (PIL.Image): Original image
        ripeness_level (str): Target ripeness level (unripe, ripe, overripe, spoiled)
        
    Returns:
        PIL.Image: Modified image simulating the specified ripeness level
    """
    # Input validation
    if not isinstance(image, Image.Image):
        raise TypeError("Input must be a PIL Image")
    
    if ripeness_level not in ["unripe", "ripe", "overripe", "spoiled"]:
        raise ValueError("Ripeness level must be one of: unripe, ripe, overripe, spoiled")
    
    # Make a copy to avoid modifying the original
    img = image.copy()
    
    # Convert to RGB mode to ensure color processing works correctly
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Different transformations based on ripeness level
    if ripeness_level == "unripe":
        # Adjust color balance toward green for unripe fruits
        # Increase green channel, slightly decrease red channel
        r, g, b = img.split()
        
        # Enhance green channel
        g = ImageEnhance.Brightness(g).enhance(1.2)
        
        # Reduce red channel slightly
        r = ImageEnhance.Brightness(r).enhance(0.85)
        
        # Recombine channels
        img = Image.merge("RGB", (r, g, b))
        
        # Slight increase in overall brightness
        img = ImageEnhance.Brightness(img).enhance(1.1)
        
        # Slight increase in contrast
        img = ImageEnhance.Contrast(img).enhance(1.1)
        
    elif ripeness_level == "ripe":
        # For ripe fruits, use original "fresh" images with slight enhancement
        # Enhance color saturation slightly
        img = ImageEnhance.Color(img).enhance(1.1)
        
        # Slight increase in contrast
        img = ImageEnhance.Contrast(img).enhance(1.1)
        
    elif ripeness_level == "overripe":
        # For overripe fruits, adjust color and texture
        # Reduce saturation slightly
        img = ImageEnhance.Color(img).enhance(0.9)
        
        # Add slight yellow/brown tint
        np_img = np.array(img)
        
        # Increase red and green channels slightly to create yellow/brown tint
        np_img[:,:,0] = np.clip(np_img[:,:,0] * 1.05, 0, 255)  # Red channel
        np_img[:,:,1] = np.clip(np_img[:,:,1] * 1.02, 0, 255)  # Green channel
        
        # Convert back to PIL Image
        img = Image.fromarray(np_img.astype('uint8'))
        
        # Add subtle texture changes
        img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=50, threshold=3))
        
        # Add small random dark spots to simulate beginning of spoilage
        np_img = np.array(img)
        height, width = np_img.shape[:2]
        
        # Add 2-5 random dark spots
        num_spots = random.randint(2, 5)
        for _ in range(num_spots):
            # Random spot location
            center_x = random.randint(0, width-1)
            center_y = random.randint(0, height-1)
            
            # Random spot size
            spot_radius = random.randint(3, 10)
            
            # Create a circular mask
            y, x = np.ogrid[:height, :width]
            dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            mask = dist_from_center <= spot_radius
            
            # Apply darkening to the spot area
            darkening_factor = random.uniform(0.7, 0.9)
            np_img[mask] = (np_img[mask] * darkening_factor).astype('uint8')
        
        # Convert back to PIL Image
        img = Image.fromarray(np_img)
        
    elif ripeness_level == "spoiled":
        # For spoiled fruits, use enhanced versions of "rotten" images
        # Reduce saturation significantly
        img = ImageEnhance.Color(img).enhance(0.7)
        
        # Add brown/dark tint
        np_img = np.array(img)
        
        # Increase red channel, decrease blue channel to create brown tint
        np_img[:,:,0] = np.clip(np_img[:,:,0] * 1.1, 0, 255)  # Red channel
        np_img[:,:,2] = np.clip(np_img[:,:,2] * 0.85, 0, 255)  # Blue channel
        
        # Convert back to PIL Image
        img = Image.fromarray(np_img.astype('uint8'))
        
        # Add texture changes for spoilage
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        
        # Add more pronounced dark/moldy spots
        np_img = np.array(img)
        height, width = np_img.shape[:2]
        
        # Add 5-10 random dark spots
        num_spots = random.randint(5, 10)
        for _ in range(num_spots):
            # Random spot location
            center_x = random.randint(0, width-1)
            center_y = random.randint(0, height-1)
            
            # Random spot size (larger than overripe)
            spot_radius = random.randint(7, 20)
            
            # Create a circular mask with fuzzy edges
            y, x = np.ogrid[:height, :width]
            dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            # Core of the spot (very dark)
            core_mask = dist_from_center <= spot_radius * 0.7
            np_img[core_mask] = (np_img[core_mask] * 0.5).astype('uint8')
            
            # Edge of the spot (moderately dark)
            edge_mask = (dist_from_center > spot_radius * 0.7) & (dist_from_center <= spot_radius)
            np_img[edge_mask] = (np_img[edge_mask] * 0.7).astype('uint8')
        
        # Convert back to PIL Image
        img = Image.fromarray(np_img)
        
        # Reduce brightness overall
        img = ImageEnhance.Brightness(img).enhance(0.9)
    
    return img

def generate_augmented_dataset(images, labels, augmentation_factor=3, balance_classes=True):
    """
    Generate an augmented dataset from original images.
    
    Args:
        images (list): List of original images (PIL.Image objects)
        labels (list): List of original labels
        augmentation_factor (int): Number of augmented versions to create per original image
        balance_classes (bool): Whether to ensure balanced distribution across categories
        
    Returns:
        tuple: (augmented_images, augmented_labels)
    """
    if len(images) != len(labels):
        raise ValueError("Number of images and labels must match")
    
    if not images:
        return [], []
    
    # Initialize lists for augmented data
    augmented_images = []
    augmented_labels = []
    
    # Add original images and labels
    augmented_images.extend(images)
    augmented_labels.extend(labels)
    
    # Get class counts for balancing
    if balance_classes:
        class_counts = {}
        for label in labels:
            if label in class_counts:
                class_counts[label] += 1
            else:
                class_counts[label] = 1
        
        # Determine the target count per class (max count * augmentation factor)
        max_count = max(class_counts.values())
        target_count = max_count * augmentation_factor
        
        # Calculate how many augmented images to create for each class
        needed_per_class = {cls: target_count - count for cls, count in class_counts.items()}
    
    # Group images by class
    images_by_class = {}
    for img, label in zip(images, labels):
        if label not in images_by_class:
            images_by_class[label] = []
        images_by_class[label].append(img)
    
    # Generate augmented images for each class
    for label, imgs in images_by_class.items():
        if balance_classes:
            # Determine how many augmented images to create for this class
            num_needed = needed_per_class[label]
            num_source = len(imgs)
            # Each source image will be used approximately this many times
            augmentations_per_image = num_needed // num_source + 1
            
            # Create and add augmented images until we have enough
            added_count = 0
            while added_count < num_needed:
                # Choose a random source image
                source_img = random.choice(imgs)
                
                # Create an augmented version with random transformations
                aug_level = random.uniform(0.3, 0.7)  # Random augmentation strength
                aug_img = augment_image(source_img, aug_level)
                
                augmented_images.append(aug_img)
                augmented_labels.append(label)
                added_count += 1
                
                # Break if we've reached the target
                if added_count >= num_needed:
                    break
        else:
            # Simple approach: create a fixed number of augmented versions per image
            for img in imgs:
                for _ in range(augmentation_factor):
                    aug_level = random.uniform(0.3, 0.7)
                    aug_img = augment_image(img, aug_level)
                    
                    augmented_images.append(aug_img)
                    augmented_labels.append(label)
    
    logger.info(f"Generated augmented dataset: {len(augmented_images)} images from {len(images)} originals")
    
    # Shuffle the dataset to mix classes
    combined = list(zip(augmented_images, augmented_labels))
    random.shuffle(combined)
    augmented_images, augmented_labels = zip(*combined)
    
    return list(augmented_images), list(augmented_labels)

def expand_binary_to_multiple_classes(images, binary_labels, target_classes=["unripe", "ripe", "overripe", "spoiled"]):
    """
    Expand a binary classification dataset (fresh/rotten) to multiple ripeness levels.
    
    Args:
        images (list): List of original images (PIL.Image objects)
        binary_labels (list): List of binary labels (0=fresh, 1=rotten)
        target_classes (list): List of target ripeness categories
        
    Returns:
        tuple: (expanded_images, expanded_labels)
    """
    if len(target_classes) != 4:
        raise ValueError("Expected exactly 4 target classes")
    
    if len(images) != len(binary_labels):
        raise ValueError("Number of images and labels must match")
    
    # Map binary labels to target classes
    # 0 (fresh) -> unripe, ripe
    # 1 (rotten) -> overripe, spoiled
    binary_to_multi = {
        0: [target_classes[0], target_classes[1]],          # fresh -> unripe, ripe
        1: [target_classes[2], target_classes[3]]           # rotten -> overripe, spoiled
    }
    
    expanded_images = []
    expanded_labels = []
    
    # Process each image in the dataset
    for img, binary_label in zip(images, binary_labels):
        # Original image with its appropriate new label
        if binary_label == 0:  # Fresh
            # Original fresh image gets labeled as "ripe"
            expanded_images.append(img)
            expanded_labels.append(target_classes[1])  # "ripe"
            
            # Create unripe version
            unripe_img = create_ripeness_gradient(img, target_classes[0])
            expanded_images.append(unripe_img)
            expanded_labels.append(target_classes[0])  # "unripe"
        else:  # Rotten
            # Original rotten image gets labeled as "spoiled"
            expanded_images.append(img)
            expanded_labels.append(target_classes[3])  # "spoiled"
            
            # Create overripe version
            overripe_img = create_ripeness_gradient(img, target_classes[2])
            expanded_images.append(overripe_img)
            expanded_labels.append(target_classes[2])  # "overripe"
    
    logger.info(f"Expanded binary dataset to {len(expanded_images)} images across {len(target_classes)} classes")
    
    return expanded_images, expanded_labels
