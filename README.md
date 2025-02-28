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

Prerequisites

Before you begin, ensure you have the following installed:

    Python: Python 3.7 or higher is required. You can download it from the official Python website: https://www.python.org/downloads/

    Git: Git is used to clone the repository. You can download it from: https://git-scm.com/downloads
Installation

Follow these steps to get the application up and running:

    Clone the Repository

    Open your terminal (macOS) or command prompt (Windows) and navigate to the directory where you want to store the application. Then, clone the repository using the following command:

bash
git clone https://github.com/imbriankeeley/Python-Capstone.git
cd Python-Capstone

Create a Virtual Environment

It's best practice to create a virtual environment to isolate the project dependencies.

    Windows:

bash
python -m venv venv
venv\Scripts\activate

macOS:

    bash
    python3 -m venv venv
    source venv/bin/activate

Install Dependencies

Install the required Python packages using pip:

    bash
    pip install -r requirements.txt

Running the Application

    Model Training (Optional)

    The model is already pre-trained, so this step is generally not necessary. However, if you wish to retrain the model, you can run:

bash
python train_model.py

Run the Application

Start the application using the following command:

    bash
    python main.py

    This will start the application. Follow the on-screen instructions or refer to the application's specific documentation for usage details.

Additional Notes

    Troubleshooting: If you encounter any issues, double-check that you have activated the virtual environment and that all dependencies are installed correctly.


### Windows
```bash
# Clone the repository
git clone https://github.com/imbriankeeley/Python-Capstone.git
cd Python-Capstone

# Create a vm
python -m venv venv
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt

# Train model (not necessary: model already trained)
python train_model.py

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

## License

[MIT License](LICENSE)
