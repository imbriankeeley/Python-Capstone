"""
Fruit Ripeness Classification System - Main Application Entry Point

This module serves as the entry point for the Fruit Ripeness Classification System,
a standalone application for assessing fruit ripeness using computer vision.
"""

import os
import sys
import tkinter as tk
import logging
from ui.main_window import FruitRipenessApp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)

def check_dependencies():
    """
    Check that all required dependencies are installed.
    Returns True if all dependencies are met, False otherwise.
    """
    try:
        import numpy
        import cv2
        import PIL
        from PIL import Image, ImageTk
        import matplotlib
        matplotlib.use('Agg')  # Set non-interactive backend for matplotlib
        import matplotlib.pyplot as plt
        import seaborn as sns
        import sklearn
        
        logger.info("All dependencies successfully loaded.")
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {str(e)}")
        return False

def setup_application():
    """
    Set up the application directories and resources.
    """
    # Create necessary directories if they don't exist
    os.makedirs('output', exist_ok=True)
    os.makedirs('temp', exist_ok=True)

def main():
    """
    Main application entry point.
    """
    logger.info("Starting Fruit Ripeness Classification System")
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Critical dependencies are missing. Please install required packages.")
        sys.exit(1)
    
    # Setup application resources
    setup_application()
    
    # Create and run the application
    app = FruitRipenessApp()
    app.run()
    
    logger.info("Application closed")

if __name__ == "__main__":
    main()
