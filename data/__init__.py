"""
Data handling package for the Fruit Ripeness Classification System.

File path: /data/__init__.py
"""

from data.data_loader import load_image, load_training_dataset
from data.data_augmentation import augment_image, create_ripeness_gradient
