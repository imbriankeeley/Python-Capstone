"""
Main application window for the Fruit Ripeness Classification System.

This module implements the main application window and coordinates
the interaction between different UI components.
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import logging
from PIL import Image, ImageTk
import numpy as np
import threading

from ui.image_upload import ImageUploadFrame
from ui.results_display import ResultsDisplayFrame
from ui.visualization import VisualizationFrame
from models.classifier import FruitRipenessClassifier
from data.data_loader import load_image
from utils.image_processing import extract_image_for_display

logger = logging.getLogger(__name__)

class MainWindow:
    """
    Main application window containing all UI components.
    """
    
    def __init__(self, root):
        """
        Initialize the main application window.
        
        Args:
            root (tk.Tk): Root Tkinter window
        """
        self.root = root
        self.root.title("Fruit Ripeness Classification System")
        self.root.geometry("1024x768")
        self.root.minsize(800, 600)
        
        # Initialize the classifier
        self.classifier = FruitRipenessClassifier()
        
        # Try to load pre-trained model
        self._load_model()
        
        # Setup the UI components
        self._setup_ui()
        
        # Set up status variables
        self.current_image_path = None
        self.current_classification_results = None
        
        # Create a queue for background processing
        self.processing = False
    
    def _load_model(self):
        """
        Load a pre-trained model if available.
        """
        model_path = os.path.join('models', 'trained_model.pkl')
        
        if os.path.exists(model_path):
            logger.info(f"Loading pre-trained model from {model_path}")
            success = self.classifier.load_model(model_path)
            if success:
                logger.info("Successfully loaded pre-trained model")
            else:
                logger.warning("Failed to load pre-trained model")
        else:
            logger.info("No pre-trained model found")
    
    def _setup_ui(self):
        """
        Set up the UI components for the main window.
        """
        # Create main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Configure grid layout for the main frame
        self.main_frame.columnconfigure(0, weight=1)  # Left column (image upload)
        self.main_frame.columnconfigure(1, weight=2)  # Right column (results + visualization)
        self.main_frame.rowconfigure(0, weight=1)  # Single row for content
        
        # Create left panel for image upload
        self.left_panel = ttk.Frame(self.main_frame)
        self.left_panel.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        # Create right panel for results and visualization
        self.right_panel = ttk.Frame(self.main_frame)
        self.right_panel.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        
        # Configure right panel grid
        self.right_panel.columnconfigure(0, weight=1)
        self.right_panel.rowconfigure(0, weight=1)  # Results
        self.right_panel.rowconfigure(1, weight=2)  # Visualization
        
        # Create and configure image upload component
        self.image_upload = ImageUploadFrame(self.left_panel, self.process_image)
        self.image_upload.pack(fill=tk.BOTH, expand=True)
        
        # Create and configure results display component
        self.results_display = ResultsDisplayFrame(self.right_panel)
        self.results_display.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        # Create and configure visualization component
        self.visualization = VisualizationFrame(self.right_panel)
        self.visualization.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        
        # Add status bar at the bottom
        self.status_bar = ttk.Frame(self.root, relief=tk.SUNKEN, padding=(2, 2))
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_text = tk.StringVar()
        self.status_text.set("Ready")
        self.status_label = ttk.Label(self.status_bar, textvariable=self.status_text, anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Add menu bar
        self._setup_menu()
    
    def _setup_menu(self):
        """
        Set up the application menu.
        """
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Image", command=self._open_image)
        file_menu.add_command(label="Save Results", command=self._save_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Model menu
        model_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Model", menu=model_menu)
        model_menu.add_command(label="Load Model", command=self._load_model_dialog)
        model_menu.add_command(label="Save Model", command=self._save_model_dialog)
        model_menu.add_separator()
        model_menu.add_command(label="Train Model", command=self._train_model_dialog)
        model_menu.add_command(label="Evaluate Model", command=self._evaluate_model_dialog)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)
        help_menu.add_command(label="Instructions", command=self._show_instructions)
    
    def process_image(self, image_path):
        """
        Process the uploaded image and update results.
        
        Args:
            image_path (str): Path to the uploaded image
        """
        if self.processing:
            messagebox.showinfo("Processing in Progress", "Please wait for the current processing to complete.")
            return
        
        self.current_image_path = image_path
        
        if not self.classifier.is_trained:
            messagebox.showwarning("Model Not Trained", 
                                  "The classifier model is not trained. Classification results may not be accurate.")
        
        self.processing = True
        self.status_text.set("Processing image...")
        
        # Start the processing in a separate thread to keep the UI responsive
        threading.Thread(target=self._process_image_thread, args=(image_path,), daemon=True).start()
    
    def _process_image_thread(self, image_path):
        """
        Thread function for processing images without blocking the UI.
        
        Args:
            image_path (str): Path to the uploaded image
        """
        try:
            # Load and preprocess the image
            processed_image = load_image(image_path)
            
            # Classify the image using the model
            results = self.classifier.predict(processed_image)
            
            if not results["success"]:
                self.root.after(0, lambda: messagebox.showerror("Classification Error", results["error"]))
                self.root.after(0, lambda: self.status_text.set("Classification failed"))
                self.processing = False
                return
            
            # Store the results
            self.current_classification_results = results
            
            # Update the UI in the main thread
            self.root.after(0, lambda: self._update_ui_with_results(results, image_path))
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            self.root.after(0, lambda: messagebox.showerror("Processing Error", f"Error processing image: {str(e)}"))
            self.root.after(0, lambda: self.status_text.set("Error processing image"))
            
        finally:
            self.processing = False
    
    def _update_ui_with_results(self, results, image_path):
        """
        Update the UI with classification results.
        
        Args:
            results (dict): Classification results
            image_path (str): Path to the processed image
        """
        # Update the results display
        self.results_display.update_results(results)
        
        # Update visualizations
        self.visualization.update_visualizations(results, image_path)
        
        # Update status
        self.status_text.set(f"Classification complete: {results['predicted_class']} ({results['confidence']:.2f})")
        
        logger.info(f"Processed image: {image_path}, Result: {results['predicted_class']}")
    
    def _open_image(self):
        """
        Open an image file dialog.
        """
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        
        if file_path:
            self.process_image(file_path)
    
    def _save_results(self):
        """
        Save the current results to a file.
        """
        if not self.current_classification_results:
            messagebox.showinfo("No Results", "No classification results to save.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'w') as f:
                f.write("Fruit Ripeness Classification Results\n")
                f.write("===================================\n\n")
                
                if self.current_image_path:
                    f.write(f"Image: {os.path.basename(self.current_image_path)}\n\n")
                
                results = self.current_classification_results
                f.write(f"Classification: {results['predicted_class']}\n")
                f.write(f"Confidence: {results['confidence']:.4f}\n\n")
                
                f.write("Confidence Scores:\n")
                for class_name, score in results['confidence_scores'].items():
                    f.write(f"  {class_name}: {score:.4f}\n")
                
                f.write("\nRecommendation:\n")
                f.write(f"  Action: {results['recommendation']['action']}\n")
                f.write(f"  Description: {results['recommendation']['description']}\n")
                
                if results['recommendation'].get('confidence_note'):
                    f.write(f"  Note: {results['recommendation']['confidence_note']}\n")
            
            messagebox.showinfo("Save Successful", "Results saved successfully.")
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Error saving results: {str(e)}")
    
    def _load_model_dialog(self):
        """
        Open a dialog to load a saved model.
        """
        file_path = filedialog.askopenfilename(
            title="Load Model",
            filetypes=[("Model files", "*.pkl"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        success = self.classifier.load_model(file_path)
        
        if success:
            messagebox.showinfo("Model Loaded", "Model loaded successfully.")
            self.status_text.set("Model loaded")
        else:
            messagebox.showerror("Load Error", "Error loading model.")
    
    def _save_model_dialog(self):
        """
        Open a dialog to save the current model.
        """
        if not self.classifier.is_trained:
            messagebox.showwarning("Model Not Trained", "Cannot save an untrained model.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Model",
            defaultextension=".pkl",
            filetypes=[("Model files", "*.pkl"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        success = self.classifier.save_model(file_path)
        
        if success:
            messagebox.showinfo("Model Saved", "Model saved successfully.")
        else:
            messagebox.showerror("Save Error", "Error saving model.")
    
    def _train_model_dialog(self):
        """
        Open a dialog to train the model.
        """
        # This would normally open a more complex dialog for selecting training data
        # For now, we'll keep it simple
        messagebox.showinfo("Training Not Implemented", 
                           "Model training functionality requires a dataset. This would typically open a dialog to select training data.")
    
    def _evaluate_model_dialog(self):
        """
        Open a dialog to evaluate the model.
        """
        # This would normally open a more complex dialog for model evaluation
        messagebox.showinfo("Evaluation Not Implemented", 
                           "Model evaluation functionality requires a test dataset. This would typically open a dialog to select test data.")
    
    def _show_about(self):
        """
        Show the about dialog.
        """
        about_text = """
        Fruit Ripeness Classification System
        
        Version 1.0
        
        A computer vision application for assessing fruit ripeness
        to reduce food waste and optimize inventory management.
        
        This application classifies fruits into four ripeness categories:
        unripe, ripe, overripe, and spoiled.
        """
        
        messagebox.showinfo("About", about_text)
    
    def _show_instructions(self):
        """
        Show the instructions dialog.
        """
        instructions_text = """
        How to use the Fruit Ripeness Classification System:
        
        1. Upload an image of a fruit using the 'Upload' button or File -> Open Image
        2. The system will automatically classify the fruit ripeness
        3. View the classification results and recommendation
        4. Explore the visualizations for more insights
        5. Save the results using File -> Save Results
        
        For best results:
        - Use well-lit images with the fruit as the main subject
        - Minimize background distractions
        - Capture the fruit from a neutral angle
        """
        
        messagebox.showinfo("Instructions", instructions_text)
