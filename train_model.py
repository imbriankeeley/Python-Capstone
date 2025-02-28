"""
Training script for the Fruit Ripeness Classification System.

This script trains the classification model using the Kaggle dataset
and saves the trained model for use by the application.
"""

import os
import logging
import numpy as np
from PIL import Image
import glob
from sklearn.model_selection import train_test_split

from models.classifier import FruitRipenessClassifier
from data.data_loader import load_image
from data.data_augmentation import expand_binary_to_multiple_classes, generate_augmented_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def load_kaggle_dataset(dataset_dir):
    """
    Load images from the Kaggle Fruits Fresh and Rotten dataset.
    
    Args:
        dataset_dir (str): Path to the dataset directory
        
    Returns:
        tuple: (images, labels)
    """
    images = []
    binary_labels = []  # 0 for fresh, 1 for rotten
    
    # Define the category mappings
    categories = {
        'freshapples': 0, 'freshbanana': 0, 'freshoranges': 0,
        'rottenapples': 1, 'rottenbanana': 1, 'rottenoranges': 1
    }
    
    logger.info(f"Loading Kaggle dataset from {dataset_dir}")
    
    # Iterate through each category
    for category, label in categories.items():
        category_path = os.path.join(dataset_dir, category)
        
        if not os.path.exists(category_path):
            logger.warning(f"Category directory not found: {category_path}")
            continue
        
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(glob.glob(os.path.join(category_path, ext)))
        
        logger.info(f"Found {len(image_files)} images in category {category}")
        
        # Load a subset of images for faster training
        max_images = 100  # Adjust as needed
        selected_files = image_files[:max_images]
        
        for img_path in selected_files:
            try:
                # Load and preprocess the image
                img = Image.open(img_path)
                img = img.resize((224, 224))  # Resize for consistency
                
                images.append(img)
                binary_labels.append(label)
            except Exception as e:
                logger.warning(f"Error loading image {img_path}: {str(e)}")
                continue
    
    logger.info(f"Successfully loaded {len(images)} images")
    return images, binary_labels

def main():
    # Paths
    dataset_dir = 'datasets/fruits/train'
    model_output_dir = 'models'
    model_file = os.path.join(model_output_dir, 'trained_model.pkl')
    
    # Create output directory if it doesn't exist
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Load the dataset
    images, binary_labels = load_kaggle_dataset(dataset_dir)
    
    if not images:
        logger.error("No images loaded. Check dataset path.")
        return
    
    # Expand binary labels to four ripeness categories
    logger.info("Expanding binary classification to four ripeness categories")
    expanded_images, expanded_labels = expand_binary_to_multiple_classes(images, binary_labels)
    
    # Generate augmented dataset for better training
    logger.info("Generating augmented dataset")
    augmented_images, augmented_labels = generate_augmented_dataset(
        expanded_images, expanded_labels, 
        augmentation_factor=2, 
        balance_classes=True
    )
    
    # Convert labels to numeric indices
    ripeness_levels = ["unripe", "ripe", "overripe", "spoiled"]
    label_indices = [ripeness_levels.index(label) for label in augmented_labels]
    
    # Convert images to numpy arrays for feature extraction
    logger.info("Converting images to numpy arrays")
    image_arrays = []
    for img in augmented_images:
        img_array = np.array(img)
        # Ensure image has 3 channels (RGB)
        if len(img_array.shape) == 2:
            img_array = np.stack((img_array,) * 3, axis=-1)
        elif img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        
        image_arrays.append(img_array)
    
    # Split into training and testing sets
    logger.info("Splitting data into training and testing sets")
    X_train, X_test, y_train, y_test = train_test_split(
        image_arrays, label_indices, 
        test_size=0.2, 
        random_state=42, 
        stratify=label_indices
    )
    
    # Initialize and train the classifier
    logger.info("Training the classifier")
    classifier = FruitRipenessClassifier()
    training_result = classifier.train(X_train, y_train)
    
    if training_result["success"]:
        logger.info(f"Training completed with accuracy: {training_result['accuracy']:.4f}")
        
        # Evaluate the classifier
        logger.info("Evaluating the classifier")
        evaluation_result = classifier.evaluate(X_test, y_test)
        
        if evaluation_result["success"]:
            logger.info(f"Evaluation accuracy: {evaluation_result['accuracy']:.4f}")
        else:
            logger.error(f"Evaluation failed: {evaluation_result.get('error')}")
        
        # Save the trained model
        logger.info(f"Saving trained model to {model_file}")
        if classifier.save_model(model_file):
            logger.info("Model saved successfully")
        else:
            logger.error("Failed to save model")
    else:
        logger.error(f"Training failed: {training_result.get('error')}")

if __name__ == "__main__":
    main()
