"""
Image upload component for the Fruit Ripeness Classification System.

File path: /ui/image_upload.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import logging

logger = logging.getLogger(__name__)

class ImageUploadFrame(ttk.Frame):
    """
    Frame for uploading and displaying input images.
    """
    
    def __init__(self, parent, image_callback):
        """
        Initialize the image upload frame.
        
        Args:
            parent: Parent tkinter container
            image_callback: Callback function to be called when an image is uploaded
        """
        super().__init__(parent)
        self.parent = parent
        self.image_callback = image_callback
        self.image_path = None
        self.photo_image = None  # Keep a reference to avoid garbage collection
        
        # Define constants
        self.PREVIEW_MAX_SIZE = (350, 350)
        self.ALLOWED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
        
        # Setup UI components
        self._setup_ui()
        
    def _setup_ui(self):
        """
        Set up the UI components for the image upload frame.
        """
        # Configure frame
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=0)  # Instructions
        self.rowconfigure(1, weight=0)  # Upload button
        self.rowconfigure(2, weight=1)  # Image preview
        self.rowconfigure(3, weight=0)  # Image info
        
        # Instructions label
        instructions_text = "Upload a fruit image for ripeness classification"
        self.instructions_label = ttk.Label(
            self, 
            text=instructions_text,
            font=('Helvetica', 11),
            anchor='center',
            padding=(0, 10)
        )
        self.instructions_label.grid(row=0, column=0, sticky='ew')
        
        # Upload button
        self.button_frame = ttk.Frame(self)
        self.button_frame.grid(row=1, column=0, pady=10, sticky='ew')
        self.button_frame.columnconfigure(0, weight=1)
        
        self.upload_button = ttk.Button(
            self.button_frame,
            text="Upload Image",
            command=self._upload_image,
            padding=(20, 5)
        )
        self.upload_button.grid(row=0, column=0)
        
        # Image preview frame with border
        self.preview_frame = ttk.LabelFrame(self, text="Image Preview", padding=(10, 5))
        self.preview_frame.grid(row=2, column=0, sticky='nsew', padx=10, pady=5)
        self.preview_frame.columnconfigure(0, weight=1)
        self.preview_frame.rowconfigure(0, weight=1)
        
        # Create a canvas for the image preview
        self.canvas = tk.Canvas(
            self.preview_frame,
            background='#f0f0f0',
            relief=tk.SUNKEN,
            borderwidth=1,
            highlightthickness=0
        )
        self.canvas.grid(row=0, column=0, sticky='nsew')
        
        # Default message in the canvas
        self.canvas.create_text(
            self.PREVIEW_MAX_SIZE[0] // 2,
            self.PREVIEW_MAX_SIZE[1] // 2,
            text="No image uploaded",
            fill='gray',
            font=('Helvetica', 10, 'italic')
        )
        self.canvas.config(width=self.PREVIEW_MAX_SIZE[0], height=self.PREVIEW_MAX_SIZE[1])
        
        # Image info label
        self.image_info = tk.StringVar()
        self.image_info.set("No image selected")
        self.info_label = ttk.Label(
            self,
            textvariable=self.image_info,
            anchor='center',
            font=('Helvetica', 9),
            foreground='gray'
        )
        self.info_label.grid(row=3, column=0, sticky='ew', pady=(5, 10))
        
    def _upload_image(self):
        """
        Handle the image upload process.
        """
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", " ".join(f"*{ext}" for ext in self.ALLOWED_EXTENSIONS)),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            # User canceled the dialog
            return
        
        # Check if the selected file is an image with allowed extension
        _, file_ext = os.path.splitext(file_path.lower())
        if file_ext not in self.ALLOWED_EXTENSIONS:
            logger.warning(f"Unsupported file format: {file_ext}")
            messagebox.showwarning(
                "Unsupported Format",
                f"The selected file format is not supported.\n"
                f"Please use one of: {', '.join(self.ALLOWED_EXTENSIONS)}"
            )
            return
        
        try:
            # Display the selected image
            self._display_image(file_path)
            
            # Call the callback function with the image path
            self.image_callback(file_path)
            
            # Update the status
            self.image_path = file_path
            logger.info(f"Image uploaded: {file_path}")
            
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            messagebox.showerror(
                "Image Error",
                f"Could not load the selected image: {str(e)}"
            )
        

    def _display_image(self, image_path):
        """
        Display the selected image in the preview area.

        Args:
            image_path (str): Path to the image to display
        """
        try:
            # Clear previous image and messages
            self.canvas.delete("all")

            # Print debug info about the path
            logger.debug(f"Attempting to display image from path: {image_path!r}")

            # Load and resize image for display - handle errors explicitly
            try:
                image = Image.open(image_path)
            except Exception as e:
                logger.error(f"Failed to open image with PIL: {str(e)}")
                raise

            # Calculate scaling factor to fit within preview area while preserving aspect ratio
            width, height = image.size
            scale = min(
                self.PREVIEW_MAX_SIZE[0] / width,
                self.PREVIEW_MAX_SIZE[1] / height
            )

            # Calculate new dimensions
            new_width = int(width * scale)
            new_height = int(height * scale)

            # Resize the image for display
            display_image = image.resize((new_width, new_height), Image.LANCZOS)

            # Convert to PhotoImage for display in Tkinter
            try:
                self.photo_image = ImageTk.PhotoImage(display_image)
            except Exception as e:
                logger.error(f"Failed to create PhotoImage: {str(e)}")
                raise

            # Center the image in the canvas
            x_offset = (self.PREVIEW_MAX_SIZE[0] - new_width) // 2
            y_offset = (self.PREVIEW_MAX_SIZE[1] - new_height) // 2

            # Display the image on the canvas
            self.canvas.create_image(
                x_offset,
                y_offset,
                anchor='nw',
                image=self.photo_image
            )
            
            # Force the canvas to update
            self.canvas.update()

            # Update image info
            image_file = os.path.basename(image_path)
            self.image_info.set(f"File: {image_file} | Dimensions: {width} × {height}")

        except Exception as e:
            logger.error(f"Error displaying image: {str(e)}")
            # Reset the canvas with an error message
            self.canvas.delete("all")  # Make sure to clear first
            self.canvas.create_text(
                self.PREVIEW_MAX_SIZE[0] // 2,
                self.PREVIEW_MAX_SIZE[1] // 2,
                text=f"Error loading image:\n{str(e)}",
                fill='red',
                justify=tk.CENTER
            )
            raise           

    def reset(self):
        """
        Reset the upload component to its initial state.
        """
        # Clear the image path
        self.image_path = None
        
        # Clear the image preview
        self.canvas.delete("all")
        self.canvas.create_text(
            self.PREVIEW_MAX_SIZE[0] // 2,
            self.PREVIEW_MAX_SIZE[1] // 2,
            text="No image uploaded",
            fill='gray',
            font=('Helvetica', 10, 'italic')
        )
        
        # Reset image info
        self.image_info.set("No image selected")
        
        # Clear the photo_image reference
        self.photo_image = None
