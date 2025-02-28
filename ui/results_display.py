"""
Results display component for the Fruit Ripeness Classification System.

File path: /ui/results_display.py
"""

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import logging

logger = logging.getLogger(__name__)

class ResultsDisplayFrame(ttk.Frame):
    """
    Frame for displaying classification results.
    """
    
    def __init__(self, parent):
        """
        Initialize the results display frame.
        
        Args:
            parent: Parent tkinter container
        """
        super().__init__(parent)
        self.parent = parent
        
        # Define ripeness colors for visual indicators
        self.ripeness_colors = {
            "unripe": "#CCEC8A",     # Light yellow-green
            "ripe": "#69B578",        # Medium green
            "overripe": "#F0A202",    # Orange
            "spoiled": "#D62246"      # Red
        }
        
        # Define action colors for recommendations
        self.action_colors = {
            "Store": "#4B88A2",      # Blue
            "Stock": "#69B578",       # Green
            "Discount": "#F0A202",    # Orange
            "Discard": "#D62246"      # Red
        }
        
        # Setup UI components
        self._setup_ui()
        
    def _setup_ui(self):
        """
        Set up the UI components for the results display frame.
        """
        # Configure frame
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=0)  # Header
        self.rowconfigure(1, weight=0)  # Classification result
        self.rowconfigure(2, weight=0)  # Confidence scores
        self.rowconfigure(3, weight=0)  # Recommendation
        
        # Add a header label
        self.header_label = ttk.Label(
            self,
            text="Classification Results",
            font=('Helvetica', 12, 'bold'),
            anchor='center',
            padding=(0, 10)
        )
        self.header_label.grid(row=0, column=0, sticky='ew')
        
        # Classification result frame
        self.result_frame = ttk.LabelFrame(self, text="Ripeness Classification", padding=(10, 5))
        self.result_frame.grid(row=1, column=0, sticky='ew', padx=10, pady=5)
        self.result_frame.columnconfigure(0, weight=1)
        
        # Create a frame for the classification result with color indicator
        self.classification_frame = ttk.Frame(self.result_frame, padding=(5, 5))
        self.classification_frame.grid(row=0, column=0, sticky='ew')
        self.classification_frame.columnconfigure(1, weight=1)
        
        # Color indicator canvas
        self.color_indicator = tk.Canvas(
            self.classification_frame,
            width=20,
            height=20,
            highlightthickness=0
        )
        self.color_indicator.grid(row=0, column=0, padx=(0, 10))
        
        # Classification result label
        self.classification_var = tk.StringVar()
        self.classification_var.set("No classification yet")
        self.classification_label = ttk.Label(
            self.classification_frame,
            textvariable=self.classification_var,
            font=('Helvetica', 12),
            anchor='w'
        )
        self.classification_label.grid(row=0, column=1, sticky='ew')
        
        # Confidence score frame
        self.confidence_frame = ttk.LabelFrame(self, text="Confidence Scores", padding=(10, 5))
        self.confidence_frame.grid(row=2, column=0, sticky='ew', padx=10, pady=5)
        self.confidence_frame.columnconfigure(0, weight=3)  # Category name
        self.confidence_frame.columnconfigure(1, weight=1)  # Percentage
        self.confidence_frame.columnconfigure(2, weight=6)  # Progress bar
        
        # Create labels and progress bars for each ripeness category
        self.confidence_vars = {}
        self.confidence_bars = {}
        
        ripeness_categories = ["unripe", "ripe", "overripe", "spoiled"]
        for i, category in enumerate(ripeness_categories):
            # Category label
            category_label = ttk.Label(
                self.confidence_frame,
                text=category.capitalize(),
                font=('Helvetica', 10),
                anchor='w',
                padding=(5, 2)
            )
            category_label.grid(row=i, column=0, sticky='w')

            # Save a reference to the label
            category_label.name = f"{category}_label"  # Add name attribute

            # Store reference to the label in a dictionary
            if not hasattr(self, 'category_labels'):
                self.category_labels = {}
            self.category_labels[category] = category_label
            
            # Percentage variable and label
            self.confidence_vars[category] = tk.StringVar()
            self.confidence_vars[category].set("0.0%")
            
            percentage_label = ttk.Label(
                self.confidence_frame,
                textvariable=self.confidence_vars[category],
                font=('Helvetica', 10),
                anchor='e',
                padding=(0, 2)
            )
            percentage_label.grid(row=i, column=1, sticky='e', padx=(0, 5))
            
            # Progress bar for visual representation
            progress_var = tk.DoubleVar()
            progress_var.set(0.0)
            
            progress_bar = ttk.Progressbar(
                self.confidence_frame,
                variable=progress_var,
                length=200,
                mode='determinate',
                maximum=100.0
            )
            progress_bar.grid(row=i, column=2, sticky='ew', padx=5, pady=2)
            
            # Store reference to the progress bar and its variable
            self.confidence_bars[category] = (progress_bar, progress_var)
        
        # Recommendation frame
        self.recommendation_frame = ttk.LabelFrame(self, text="Recommended Action", padding=(10, 5))
        self.recommendation_frame.grid(row=3, column=0, sticky='ew', padx=10, pady=5)
        self.recommendation_frame.columnconfigure(0, weight=1)
        
        # Action label frame with background color
        self.action_frame = ttk.Frame(self.recommendation_frame, padding=(5, 5))
        self.action_frame.grid(row=0, column=0, sticky='ew')
        self.action_frame.columnconfigure(1, weight=1)
        
        # Action color indicator
        self.action_indicator = tk.Canvas(
            self.action_frame,
            width=20,
            height=20,
            highlightthickness=0
        )
        self.action_indicator.grid(row=0, column=0, padx=(0, 10))
        
        # Action label
        self.action_var = tk.StringVar()
        self.action_var.set("No recommendation yet")
        self.action_label = ttk.Label(
            self.action_frame,
            textvariable=self.action_var,
            font=('Helvetica', 12, 'bold'),
            anchor='w'
        )
        self.action_label.grid(row=0, column=1, sticky='ew')
        
        # Description label
        self.description_var = tk.StringVar()
        self.description_label = ttk.Label(
            self.recommendation_frame,
            textvariable=self.description_var,
            wraplength=300,
            font=('Helvetica', 10),
            anchor='w',
            padding=(5, 5)
        )
        self.description_label.grid(row=1, column=0, sticky='ew')
        
        # Confidence note label (hidden by default)
        self.confidence_note_var = tk.StringVar()
        self.confidence_note_label = ttk.Label(
            self.recommendation_frame,
            textvariable=self.confidence_note_var,
            wraplength=300,
            font=('Helvetica', 9, 'italic'),
            foreground='#D62246',  # Red for warning
            anchor='w',
            padding=(5, 5)
        )
        self.confidence_note_label.grid(row=2, column=0, sticky='ew')
        self.confidence_note_label.grid_remove()  # Hide initially
        
        # Initialize with empty/default state
        self.reset()
        
    def update_results(self, classification_results):
        """
        Update the display with new classification results.
        
        Args:
            classification_results (dict): Results from the classifier including:
                - predicted_class: The predicted ripeness level
                - confidence_scores: Confidence for each ripeness level
                - recommendation: Recommended action based on ripeness
        """
        if not classification_results or not classification_results.get("success", False):
            logger.warning("Cannot update results with invalid classification results")
            return
        
        # Get the predicted class and confidence
        predicted_class = classification_results.get("predicted_class", "unknown")
        confidence = classification_results.get("confidence", 0.0)
        confidence_scores = classification_results.get("confidence_scores", {})
        recommendation = classification_results.get("recommendation", {})
        
        # Update classification result
        self.classification_var.set(f"{predicted_class.capitalize()} ({confidence:.1%})")
        
        # Update color indicator
        ripeness_color = self._get_color_for_ripeness(predicted_class)
        self.color_indicator.configure(background=ripeness_color)
        self.color_indicator.create_rectangle(0, 0, 20, 20, fill=ripeness_color, outline=ripeness_color)
        
        # Update confidence scores
        for category, (_, progress_var) in self.confidence_bars.items():
            score = confidence_scores.get(category, 0.0)
            percentage = score * 100.0
            
            self.confidence_vars[category].set(f"{percentage:.1f}%")
            progress_var.set(percentage)
            
            # Highlight the predicted class in the confidence scores
            if category == predicted_class:
                # Use the stored reference instead of nametowidget
                if category in self.category_labels:
                    self.category_labels[category].configure(font=('Helvetica', 10, 'bold'))
            else:
                if category in self.category_labels:
                    self.category_labels[category].configure(font=('Helvetica', 10))
        
        # Update recommendation
        action = recommendation.get("action", "Unknown")
        description = recommendation.get("description", "")
        confidence_note = recommendation.get("confidence_note", "")
        
        self.action_var.set(action)
        self.description_var.set(description)
        
        # Update action color indicator
        action_color = self._get_color_for_action(action)
        self.action_indicator.configure(background=action_color)
        self.action_indicator.create_rectangle(0, 0, 20, 20, fill=action_color, outline=action_color)
        
        # Show or hide confidence note
        if confidence_note:
            self.confidence_note_var.set(confidence_note)
            self.confidence_note_label.grid()
        else:
            self.confidence_note_label.grid_remove()
        
        logger.debug(f"Updated results display with classification: {predicted_class}")
        
    def _get_color_for_ripeness(self, ripeness_level):
        """
        Get the display color for a ripeness level.
        
        Args:
            ripeness_level (str): Ripeness level
            
        Returns:
            str: Hex color code
        """
        return self.ripeness_colors.get(ripeness_level.lower(), "#CCCCCC")  # Default gray if not found
        
    def _get_color_for_action(self, action):
        """
        Get the display color for a recommendation action.
        
        Args:
            action (str): Recommended action
            
        Returns:
            str: Hex color code
        """
        return self.action_colors.get(action, "#CCCCCC")  # Default gray if not found
        
    def reset(self):
        """
        Reset the display to its initial state.
        """
        # Reset classification
        self.classification_var.set("No classification yet")
        self.color_indicator.configure(background="#CCCCCC")
        self.color_indicator.create_rectangle(0, 0, 20, 20, fill="#CCCCCC", outline="#CCCCCC")
        
        # Reset confidence scores
        for category, (_, progress_var) in self.confidence_bars.items():
            self.confidence_vars[category].set("0.0%")
            progress_var.set(0.0)
        
        # Reset recommendation
        self.action_var.set("No recommendation yet")
        self.description_var.set("")
        self.confidence_note_var.set("")
        self.confidence_note_label.grid_remove()
        self.action_indicator.configure(background="#CCCCCC")
        self.action_indicator.create_rectangle(0, 0, 20, 20, fill="#CCCCCC", outline="#CCCCCC")
        
        logger.debug("Reset results display")
