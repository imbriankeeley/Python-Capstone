"""
Visualization components for the Fruit Ripeness Classification System.

This module implements the visualization components for displaying
classification results, including confusion matrix, feature heatmap,
and confidence distribution plots.

File path: /ui/visualization.py
"""

import os
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns
import logging
from PIL import Image

from models.model_utils import get_importance_map
from utils.visualization_utils import (
    create_confusion_matrix_figure,
    create_feature_heatmap_figure,
    create_distribution_plot_figure,
    save_figure
)
from utils.image_processing import preprocess_image, normalize_image
from data.data_loader import load_image

logger = logging.getLogger(__name__)

class VisualizationFrame(ttk.Frame):
    """
    Frame for displaying visualizations of classification results.
    """
    
    def __init__(self, parent):
        """
        Initialize the visualization frame.
        
        Args:
            parent: Parent tkinter container
        """
        super().__init__(parent)
        self.parent = parent
        
        # Store references to figures and canvases to prevent garbage collection
        self.figures = {}
        self.canvases = {}
        
        # Store current visualization data
        self.current_data = {
            "confusion_matrix": None,
            "feature_heatmap": None,
            "distribution_plot": None
        }
        
        # Define constants
        self.TAB_NAMES = ["Confidence Distribution", "Feature Heatmap", "Confusion Matrix"]
        self.FIGURE_SIZE = (5, 4)
        self.DPI = 100
        
        # Setup UI components
        self._setup_ui()
        
    def _setup_ui(self):
        """
        Set up the UI components for the visualization frame.
        """
        # Configure frame
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=0)  # Header
        self.rowconfigure(1, weight=1)  # Tabs
        
        # Create header label
        header_label = ttk.Label(
            self,
            text="Visualizations",
            font=('Helvetica', 12, 'bold'),
            anchor='center',
            padding=(0, 10)
        )
        header_label.grid(row=0, column=0, sticky='ew')
        
        # Create tabbed interface
        self.tab_control = ttk.Notebook(self)
        self.tab_control.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)
        
        # Create tabs for different visualizations
        self.tabs = {}
        
        for tab_name in self.TAB_NAMES:
            tab = ttk.Frame(self.tab_control)
            tab.columnconfigure(0, weight=1)
            tab.rowconfigure(0, weight=1)  # Canvas area
            tab.rowconfigure(1, weight=0)  # Button area
            
            self.tab_control.add(tab, text=tab_name)
            self.tabs[tab_name] = tab
            
            # Add figure and canvas for each tab
            fig = Figure(figsize=self.FIGURE_SIZE, dpi=self.DPI)
            canvas = FigureCanvasTkAgg(fig, master=tab)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
            
            # Store references to prevent garbage collection
            self.figures[tab_name] = fig
            self.canvases[tab_name] = canvas
            
            # Add button frame for each tab
            button_frame = ttk.Frame(tab)
            button_frame.grid(row=1, column=0, sticky='ew', padx=5, pady=5)
            
            # Add save button
            save_button = ttk.Button(
                button_frame,
                text="Save Visualization",
                command=lambda t=tab_name: self._save_visualization(t)
            )
            save_button.pack(side=tk.RIGHT, padx=5)
        
        # Add placeholder text to each figure
        for tab_name, fig in self.figures.items():
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f"No data available for {tab_name}",
                   horizontalalignment='center',
                   verticalalignment='center',
                   transform=ax.transAxes,
                   fontsize=12,
                   color='gray')
            ax.set_axis_off()
            self.canvases[tab_name].draw()
        
    def update_visualizations(self, results, image_path=None):
        """
        Update all visualizations based on classification results.
        
        Args:
            results (dict): Classification results
            image_path (str, optional): Path to the classified image
        """
        if not results or not results.get("success", False):
            logger.warning("Cannot update visualizations with invalid results")
            return
        
        # Extract required data from results
        predicted_class = results.get("predicted_class", "unknown")
        confidence_scores = results.get("confidence_scores", {})
        
        # Update confidence distribution plot
        self.update_distribution_plot(confidence_scores)
        
        # Update feature heatmap if image path is provided
        if image_path and os.path.exists(image_path):
            try:
                # Load and process the image for visualization
                image = load_image(image_path)
                
                # Get importance map from the visualization utils
                # This would typically come from the classifier but we'll create a simple one
                # based on the confidence scores for this implementation
                importance_map = self._create_dummy_importance_map(image, predicted_class)
                
                self.update_feature_heatmap(image, importance_map)
            except Exception as e:
                logger.error(f"Error updating feature heatmap: {str(e)}")
        
        # Update confusion matrix with dummy data for now
        # In a real application, this would use actual confusion matrix data
        class_names = list(confidence_scores.keys())
        confusion_matrix = self._create_dummy_confusion_matrix(class_names, predicted_class)
        self.update_confusion_matrix(confusion_matrix, class_names)
        
    def update_confusion_matrix(self, confusion_matrix, class_names):
        """
        Update the confusion matrix visualization.
        
        Args:
            confusion_matrix (numpy.ndarray): Confusion matrix data
            class_names (list): Names of the classes
        """
        try:
            # Store the current data
            self.current_data["confusion_matrix"] = {
                "matrix": confusion_matrix,
                "class_names": class_names
            }
            
            # Get the figure and clear it
            fig = self.figures["Confusion Matrix"]
            fig.clear()
            
            # Create the confusion matrix visualization
            ax = fig.add_subplot(111)
            sns.heatmap(
                confusion_matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax
            )
            
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            
            # Adjust layout and draw
            fig.tight_layout()
            self.canvases["Confusion Matrix"].draw()
            
            logger.debug("Updated confusion matrix visualization")
            
        except Exception as e:
            logger.error(f"Error updating confusion matrix: {str(e)}")
            self._show_error_in_figure("Confusion Matrix", str(e))
        
    def update_feature_heatmap(self, image, importance_map):
        """
        Update the feature heatmap visualization.
        
        Args:
            image (numpy.ndarray): Original image
            importance_map (numpy.ndarray): Feature importance map
        """
        try:
            # Store the current data
            self.current_data["feature_heatmap"] = {
                "image": image,
                "importance_map": importance_map
            }
            
            # Get the figure and clear it
            fig = self.figures["Feature Heatmap"]
            fig.clear()
            
            # Create the feature heatmap visualization
            ax = fig.add_subplot(111)
            
            # If image has float values between 0-1, convert to 0-255 for display
            display_img = image.copy()
            if display_img.dtype == np.float32 or display_img.dtype == np.float64:
                if display_img.max() <= 1.0:
                    display_img = (display_img * 255).astype(np.uint8)
            
            # Show the original image
            ax.imshow(display_img)
            
            # Overlay the importance map with transparency
            cmap = plt.cm.jet
            cmap.set_bad('white', alpha=0)
            
            # Ensure importance map has same dimensions as image
            if importance_map.shape[:2] != image.shape[:2]:
                importance_map = np.resize(importance_map, image.shape[:2])
            
            # Display the importance map overlay with transparency
            im = ax.imshow(importance_map, cmap=cmap, alpha=0.5)
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Feature Importance")
            
            ax.set_title("Feature Importance Heatmap")
            ax.axis('off')
            
            # Adjust layout and draw
            fig.tight_layout()
            self.canvases["Feature Heatmap"].draw()
            
            logger.debug("Updated feature heatmap visualization")
            
        except Exception as e:
            logger.error(f"Error updating feature heatmap: {str(e)}")
            self._show_error_in_figure("Feature Heatmap", str(e))
        
    def update_distribution_plot(self, confidence_scores):
        """
        Update the confidence distribution plot.
        
        Args:
            confidence_scores (dict): Confidence scores for each class
        """
        try:
            # Store the current data
            self.current_data["distribution_plot"] = {
                "confidence_scores": confidence_scores
            }
            
            # Check if confidence scores are available
            if not confidence_scores:
                self._show_error_in_figure("Confidence Distribution", "No confidence scores available")
                return
            
            # Get class names and scores
            class_names = list(confidence_scores.keys())
            scores = [confidence_scores[name] for name in class_names]
            
            # Get the figure and clear it
            fig = self.figures["Confidence Distribution"]
            fig.clear()
            
            # Create the distribution plot
            ax = fig.add_subplot(111)
            
            # Create bar colors based on confidence (higher confidence = darker color)
            colors = plt.cm.Blues(np.array(scores) * 0.7 + 0.3)
            
            # Create the bar chart
            bars = ax.bar(class_names, scores, color=colors)
            
            # Add value annotations on top of bars
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.01,
                    f"{score:.3f}",
                    ha='center',
                    va='bottom',
                    rotation=0,
                    fontsize=9
                )
            
            # Add labels and title
            ax.set_title("Confidence Distribution Across Classes")
            ax.set_xlabel("Ripeness Category")
            ax.set_ylabel("Confidence Score")
            
            # Set y-axis to start at 0 and go to 1.0
            ax.set_ylim(0, 1.0)
            
            # Add a horizontal line at the maximum confidence
            ax.axhline(max(scores), color='red', linestyle='--', alpha=0.7)
            
            # Adjust layout and draw
            fig.tight_layout()
            self.canvases["Confidence Distribution"].draw()
            
            logger.debug("Updated confidence distribution visualization")
            
        except Exception as e:
            logger.error(f"Error updating distribution plot: {str(e)}")
            self._show_error_in_figure("Confidence Distribution", str(e))
    
    def _save_visualization(self, tab_name):
        """
        Save the current visualization to a file.
        
        Args:
            tab_name (str): Name of the tab containing the visualization to save
        """
        from tkinter import filedialog
        
        # Define default filename based on visualization type
        filename_map = {
            "Confidence Distribution": "confidence_distribution.png",
            "Feature Heatmap": "feature_heatmap.png",
            "Confusion Matrix": "confusion_matrix.png"
        }
        
        default_filename = filename_map.get(tab_name, "visualization.png")
        
        # Open file dialog
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            initialfile=default_filename
        )
        
        if not file_path:
            return
        
        try:
            # Save the figure
            fig = self.figures[tab_name]
            fig.savefig(file_path, dpi=300, bbox_inches='tight')
            
            logger.info(f"Saved {tab_name} visualization to {file_path}")
            tk.messagebox.showinfo("Save Successful", f"{tab_name} visualization saved successfully.")
            
        except Exception as e:
            logger.error(f"Error saving visualization: {str(e)}")
            tk.messagebox.showerror("Save Error", f"Error saving visualization: {str(e)}")
    
    def _show_error_in_figure(self, tab_name, error_message):
        """
        Display an error message in a figure when visualization fails.
        
        Args:
            tab_name (str): Name of the tab
            error_message (str): Error message to display
        """
        fig = self.figures[tab_name]
        fig.clear()
        
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, f"Error: {error_message}",
               horizontalalignment='center',
               verticalalignment='center',
               transform=ax.transAxes,
               fontsize=10,
               color='red',
               wrap=True)
        ax.set_axis_off()
        
        fig.tight_layout()
        self.canvases[tab_name].draw()
    
    def reset(self):
        """
        Reset all visualizations to their initial state.
        """
        # Clear all stored data
        for key in self.current_data:
            self.current_data[key] = None
        
        # Reset each figure
        for tab_name, fig in self.figures.items():
            fig.clear()
            
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f"No data available for {tab_name}",
                   horizontalalignment='center',
                   verticalalignment='center',
                   transform=ax.transAxes,
                   fontsize=12,
                   color='gray')
            ax.set_axis_off()
            
            self.canvases[tab_name].draw()
        
        logger.debug("Reset all visualizations")
    
    def _create_dummy_importance_map(self, image, predicted_class):
        """
        Create a dummy importance map for demonstration purposes.
        In a real application, this would come from model analysis.
        
        Args:
            image (numpy.ndarray): The image
            predicted_class (str): Predicted class
            
        Returns:
            numpy.ndarray: Dummy importance map
        """
        # Create a dummy importance map based on image features
        # In a real application, this would be generated by model-specific methods
        
        height, width = image.shape[:2]
        
        # Create a simple gradient importance map for demonstration
        x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
        
        # Create different patterns based on predicted class
        if predicted_class == "unripe":
            # Focus on edges (simulating focus on color transition areas)
            importance = np.sin(x * 10) * np.sin(y * 10) * 0.5 + 0.5
        
        elif predicted_class == "ripe":
            # Focus on center (simulating focus on central color)
            center_x, center_y = 0.5, 0.5
            dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            importance = 1 - (dist * 2)
            importance = np.clip(importance, 0, 1)
        
        elif predicted_class == "overripe":
            # Focus on specific areas (simulating focus on spots)
            importance = np.zeros((height, width))
            
            # Add a few "important spots"
            for _ in range(3):
                cx = np.random.rand()
                cy = np.random.rand()
                spot_size = 0.1 + 0.1 * np.random.rand()
                
                spot_dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                spot = 1 - (spot_dist / spot_size)
                spot = np.clip(spot, 0, 1)
                
                importance = np.maximum(importance, spot)
        
        elif predicted_class == "spoiled":
            # Focus on edges and spots (simulating focus on decay areas)
            edge_factor = np.sin(x * 8) * np.sin(y * 8) * 0.3 + 0.3
            
            # Add a few "important spots" for spoilage
            spots = np.zeros_like(edge_factor)
            for _ in range(5):
                cx = np.random.rand()
                cy = np.random.rand()
                spot_size = 0.05 + 0.1 * np.random.rand()
                
                spot_dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                spot = 1 - (spot_dist / spot_size)
                spot = np.clip(spot, 0, 1)
                
                spots = np.maximum(spots, spot)
            
            importance = np.maximum(edge_factor, spots)
        
        else:
            # Default importance map
            center_x, center_y = 0.5, 0.5
            dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            importance = 1 - dist
            
        # Scale importance to 0-1 range
        importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
        
        return importance
    
    def _create_dummy_confusion_matrix(self, class_names, predicted_class):
        """
        Create a dummy confusion matrix for demonstration purposes.
        In a real application, this would come from model evaluation.
        
        Args:
            class_names (list): Names of the classes
            predicted_class (str): Predicted class
            
        Returns:
            numpy.ndarray: Dummy confusion matrix
        """
        n_classes = len(class_names)
        
        # Start with an identity matrix (perfect classification)
        confusion_matrix = np.eye(n_classes, dtype=int)
        
        # Add some misclassifications to make it more realistic
        # For each row (true class), distribute some samples to incorrect classes
        for i in range(n_classes):
            # Total samples for this class
            total_samples = 100
            
            # Correctly classified samples (60-90%)
            correct_samples = int(total_samples * (0.6 + 0.3 * np.random.rand()))
            
            # Fill diagonal with correctly classified samples
            confusion_matrix[i, i] = correct_samples
            
            # Distribute remaining samples among other classes
            remaining_samples = total_samples - correct_samples
            
            # Get indices of other classes
            other_indices = [j for j in range(n_classes) if j != i]
            
            # Distribute remaining samples
            for j in range(len(other_indices)):
                if j == len(other_indices) - 1:
                    # Last index gets all remaining samples
                    confusion_matrix[i, other_indices[j]] = remaining_samples
                else:
                    # Randomly distribute some samples to this class
                    samples = int(remaining_samples * np.random.rand() * 0.7)
                    confusion_matrix[i, other_indices[j]] = samples
                    remaining_samples -= samples
        
        # Emphasize the predicted class for demonstration purposes
        # Make the predicted class have better accuracy
        predicted_index = class_names.index(predicted_class)
        row_sum = confusion_matrix[predicted_index, :].sum()
        confusion_matrix[predicted_index, :] = 0
        confusion_matrix[predicted_index, predicted_index] = int(row_sum * 0.9)
        
        # Distribute the remaining samples
        remaining = row_sum - confusion_matrix[predicted_index, predicted_index]
        for i in range(n_classes):
            if i != predicted_index:
                # Distribute remaining samples more or less evenly
                confusion_matrix[predicted_index, i] = remaining // (n_classes - 1)
        
        return confusion_matrix
