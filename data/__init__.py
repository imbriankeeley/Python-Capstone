"""
Data handling package for the Fruit Ripeness Classification System.

This package contains functions for loading and processing data,
including image loading and data augmentation techniques.

File path: /data/__init__.py
"""

from data.data_loader import (
    load_image,
    load_training_dataset,
    save_processed_image,
    get_sample_images
)

from data.data_augmentation import (
    augment_image,
    create_ripeness_gradient,
    generate_augmented_dataset,
    expand_binary_to_multiple_classes
)

__all__ = [
    # Data loading
    'load_image',
    'load_training_dataset',
    'save_processed_image',
    'get_sample_images',
    
    # Data augmentation
    'augment_image',
    'create_ripeness_gradient',
    'generate_augmented_dataset',
    'expand_binary_to_multiple_classes'
]
