"""
Utility functions for generating visualizations.

File path: /utils/visualization_utils.py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
import os
import logging

logger = logging.getLogger(__name__)

def create_confusion_matrix_figure(cm, class_names):
    """
    Create a confusion matrix visualization figure.
    
    Args:
        cm (numpy.ndarray): Confusion matrix data
        class_names (list): Names of the classes
        
    Returns:
        Figure: Matplotlib figure object
    """
    # Create a new figure
    fig = Figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111)
    
    # Create the heatmap
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    
    # Add labels and title
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    
    # Adjust layout
    fig.tight_layout()
    
    return fig

def create_feature_heatmap_figure(image, importance_map):
    """
    Create a feature importance heatmap visualization figure.
    
    Args:
        image (numpy.ndarray): Original image
        importance_map (numpy.ndarray): Feature importance map
        
    Returns:
        Figure: Matplotlib figure object
    """
    # Create a new figure
    fig = Figure(figsize=(8, 6), dpi=100)
    
    # Plot original image
    ax1 = fig.add_subplot(121)
    ax1.imshow(image)
    ax1.set_title("Original Image")
    ax1.axis('off')
    
    # Plot importance heatmap
    ax2 = fig.add_subplot(122)
    im = ax2.imshow(importance_map, cmap='jet', alpha=0.7)
    ax2.set_title("Feature Importance")
    ax2.axis('off')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax2, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Importance')
    
    # Adjust layout
    fig.tight_layout()
    
    return fig

def create_distribution_plot_figure(confidence_scores, class_names):
    """
    Create a confidence distribution plot visualization figure.
    
    Args:
        confidence_scores (dict): Confidence scores for each class
        class_names (list): Names of the classes
        
    Returns:
        Figure: Matplotlib figure object
    """
    # Create a new figure
    fig = Figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111)
    
    # Extract scores in the correct order
    scores = [confidence_scores.get(name, 0) * 100 for name in class_names]
    
    # Define colors for the bars
    colors = ['#CCEC8A', '#69B578', '#F0A202', '#D62246']  # Green to red
    
    # Create the bar chart
    bars = ax.bar(class_names, scores, color=colors)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # Add labels and title
    ax.set_xlabel("Ripeness Category")
    ax.set_ylabel("Confidence Score (%)")
    ax.set_title("Confidence Distribution")
    ax.set_ylim(0, 110)  # Leave room for the text above bars
    
    # Add gridlines for better readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout
    fig.tight_layout()
    
    return fig

def create_ripeness_comparison_figure(images, classifications):
    """
    Create a visual comparison of different ripeness levels.
    
    Args:
        images (list): List of images for different ripeness levels
        classifications (list): Corresponding classifications
        
    Returns:
        Figure: Matplotlib figure object
    """
    # Determine the number of images to display
    n_images = len(images)
    
    # Create a new figure
    fig = Figure(figsize=(10, 3), dpi=100)
    
    # Plot each image with its classification
    for i, (img, label) in enumerate(zip(images, classifications)):
        ax = fig.add_subplot(1, n_images, i+1)
        ax.imshow(img)
        ax.set_title(label.capitalize())
        ax.axis('off')
    
    # Adjust layout
    fig.tight_layout()
    
    return fig

def create_feature_importance_bar_chart(feature_importances, top_n=10):
    """
    Create a bar chart showing the top N most important features.
    
    Args:
        feature_importances (dict): Dictionary mapping feature names to importance values
        top_n (int): Number of top features to display
        
    Returns:
        Figure: Matplotlib figure object
    """
    # Sort features by importance and get top N
    if isinstance(feature_importances, dict):
        # Sort the dictionary by values in descending order
        sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
        
        # Get the top N features
        top_features = sorted_features[:top_n]
        feature_names = [item[0] for item in top_features]
        importance_values = [item[1] for item in top_features]
    else:
        # If feature_importances is not a dictionary, assume it's an array of values
        # with implicit indices as feature names
        importance_values = feature_importances[:top_n]
        feature_names = [f"Feature {i+1}" for i in range(top_n)]
    
    # Create a new figure
    fig = Figure(figsize=(10, 6), dpi=100)
    ax = fig.add_subplot(111)
    
    # Create horizontal bar chart
    y_pos = np.arange(len(feature_names))
    ax.barh(y_pos, importance_values, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    
    # Add labels and title
    ax.set_xlabel('Importance')
    ax.set_title('Top Feature Importances')
    
    # Invert y-axis to have the highest importance at the top
    ax.invert_yaxis()
    
    # Add gridlines for better readability
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    # Adjust layout
    fig.tight_layout()
    
    return fig

def save_figure(figure, file_path):
    """
    Save a figure to disk.
    
    Args:
        figure (Figure): Matplotlib figure object
        file_path (str): Path to save the figure
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save the figure
        figure.savefig(file_path, bbox_inches='tight', dpi=300)
        logger.info(f"Figure saved to {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving figure to {file_path}: {str(e)}")
        return False

def create_combined_results_figure(image, classification, confidence_scores, recommendation):
    """
    Create a combined figure with all relevant classification results.
    
    Args:
        image (numpy.ndarray): Original image
        classification (str): Predicted classification
        confidence_scores (dict): Confidence scores for each class
        recommendation (dict): Recommendation details
        
    Returns:
        Figure: Matplotlib figure object
    """
    # Create a new figure with grid layout
    fig = Figure(figsize=(12, 8), dpi=100)
    
    # Define a grid layout
    gs = fig.add_gridspec(2, 2)
    
    # Plot original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image)
    ax1.set_title("Input Image")
    ax1.axis('off')
    
    # Plot classification result
    ax2 = fig.add_subplot(gs[0, 1])
    class_names = list(confidence_scores.keys())
    scores = [confidence_scores.get(name, 0) * 100 for name in class_names]
    colors = ['#CCEC8A', '#69B578', '#F0A202', '#D62246']
    bars = ax2.bar(class_names, scores, color=colors)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # Highlight the predicted class
    predicted_index = class_names.index(classification)
    bars[predicted_index].set_edgecolor('black')
    bars[predicted_index].set_linewidth(2)
    
    ax2.set_xlabel("Ripeness Category")
    ax2.set_ylabel("Confidence Score (%)")
    ax2.set_title("Classification Results")
    ax2.set_ylim(0, 110)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Plot recommendation
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')
    
    # Create a text box for recommendation
    action = recommendation.get("action", "")
    description = recommendation.get("description", "")
    confidence_note = recommendation.get("confidence_note", "")
    
    rec_text = f"Recommendation: {action}\n\n{description}"
    if confidence_note:
        rec_text += f"\n\nNote: {confidence_note}"
    
    # Add colored box based on action
    action_colors = {
        "Store": "#4B88A2",    # Blue
        "Stock": "#69B578",    # Green
        "Discount": "#F0A202", # Orange
        "Discard": "#D62246",  # Red
        "Check": "#9370DB"     # Purple for manual check
    }
    
    box_color = action_colors.get(action, "#CCCCCC")
    rect = plt.Rectangle((0.1, 0.1), 0.8, 0.8, fill=True, color=box_color, alpha=0.2)
    ax3.add_patch(rect)
    
    # Add recommendation text
    ax3.text(0.5, 0.5, rec_text, 
             horizontalalignment='center',
             verticalalignment='center',
             fontsize=12,
             transform=ax3.transAxes,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
    
    # Adjust layout
    fig.tight_layout()
    
    return fig
