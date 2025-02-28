# Fruit Ripeness Classification System

A standalone Python application designed to help grocery store personnel quickly and accurately assess the ripeness stage of fruits using computer vision and machine learning.

## Overview

The Fruit Ripeness Classification System analyzes images of fruits and classifies them into four ripeness categories: unripe, ripe, overripe, and spoiled. This automated approach replaces manual inspection processes, with the goal of reducing food waste by 15-20% through optimized stock rotation.

## Features

- Image upload and real-time classification of fruit ripeness
- Classification into four categories: unripe, ripe, overripe, and spoiled
- Confidence scores for classification results
- Visual indicators and color-coding for quick interpretation
- Automated recommendations for inventory actions
- Visualizations including confusion matrix, feature heatmap, and distribution plot

## Technology Stack

- **Python 3.x**: Primary programming language
- **Tkinter**: GUI framework for interface development
- **Pillow (PIL)**: Image processing and manipulation
- **Matplotlib/Seaborn**: Visualization components
- **Scikit-learn**: Primary ML framework
- **OpenCV**: Image preprocessing for consistency
- **PyInstaller**: Packaging as a standalone executable

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/fruit-ripeness-classifier.git
cd Python-Capstone

# Install required dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

## Usage

1. Launch the application
2. Upload an image of a fruit using the upload button
3. View the classification result and recommended action
4. Explore visualizations for additional insights

## Development

This project is structured as follows:
- `data/`: Data loading and processing modules
- `models/`: Machine learning model implementations
- `ui/`: User interface components
- `utils/`: Utility functions
- `tests/`: Test modules

## License

[MIT License](LICENSE)
