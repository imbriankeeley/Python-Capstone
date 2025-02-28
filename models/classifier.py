"""
Main classification model for fruit ripeness detection.

This module implements the core classification model for determining
fruit ripeness levels based on image features.
"""

import os
import numpy as np
import logging
import pickle
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from models.feature_extraction import extract_features, extract_color_features, extract_texture_features

logger = logging.getLogger(__name__)

class FruitRipenessClassifier:
    """
    Classifier for determining fruit ripeness level.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the fruit ripeness classifier.
        
        Args:
            model_path (str, optional): Path to a pre-trained model file
        """
        # Initialize the classification model with default parameters
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=42
        )
        
        self.ripeness_levels = ["unripe", "ripe", "overripe", "spoiled"]
        self.is_trained = False
        self.feature_names = []
        
        # Load pre-trained model if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def train(self, training_images, training_labels):
        """
        Train the classifier on the provided dataset.
        
        Args:
            training_images (list or ndarray): List of training images
            training_labels (list or ndarray): List of corresponding labels
            
        Returns:
            dict: Training results and metrics
        """
        if len(training_images) == 0 or len(training_labels) == 0:
            logger.error("No training data provided")
            return {"success": False, "error": "No training data provided"}
        
        if len(training_images) != len(training_labels):
            logger.error("Number of images and labels must match")
            return {"success": False, "error": "Number of images and labels must match"}
        
        try:
            logger.info(f"Starting model training with {len(training_images)} images")
            start_time = time.time()
            
            # Extract features from training images
            features = []
            for image in training_images:
                image_features = extract_features(image)
                features.append(image_features)
            
            # Convert to numpy array for sklearn
            features = np.array(features)
            
            # Train the model
            self.model.fit(features, training_labels)
            self.is_trained = True
            
            # Calculate training accuracy
            training_predictions = self.model.predict(features)
            accuracy = accuracy_score(training_labels, training_predictions)
            
            # Generate training confusion matrix
            cm = confusion_matrix(training_labels, training_predictions)
            
            training_time = time.time() - start_time
            logger.info(f"Model training completed in {training_time:.2f} seconds. Accuracy: {accuracy:.4f}")
            
            # Store feature importance if available
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = self.model.feature_importances_
            else:
                feature_importance = None
            
            return {
                "success": True,
                "accuracy": accuracy,
                "confusion_matrix": cm,
                "training_time": training_time,
                "feature_importance": feature_importance
            }
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def predict(self, image):
        """
        Predict the ripeness level of a fruit image.
        
        Args:
            image (numpy.ndarray): Preprocessed fruit image
            
        Returns:
            dict: Classification results containing:
                - predicted_class: The predicted ripeness level
                - confidence_scores: Confidence for each ripeness level
                - recommendation: Recommended action based on ripeness
        """
        if not self.is_trained:
            logger.warning("Model has not been trained yet")
            return {
                "success": False,
                "error": "Model has not been trained yet"
            }
        
        try:
            # Extract features from the input image
            features = extract_features(image)
            
            # Reshape for single sample prediction if needed
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # Get prediction probabilities
            probabilities = self.model.predict_proba(features)[0]
            
            # Get the predicted class
            predicted_class_index = np.argmax(probabilities)
            predicted_class = self.ripeness_levels[predicted_class_index]
            
            # Get confidence score for the prediction (probability of the predicted class)
            confidence = probabilities[predicted_class_index]
            
            # Generate recommendation based on the prediction
            recommendation = self._generate_recommendation(predicted_class, confidence)
            
            # Prepare confidence scores for all classes
            confidence_scores = {}
            for i, level in enumerate(self.ripeness_levels):
                confidence_scores[level] = float(probabilities[i])
            
            logger.info(f"Predicted ripeness: {predicted_class} with confidence: {confidence:.4f}")
            
            return {
                "success": True,
                "predicted_class": predicted_class,
                "confidence": float(confidence),
                "confidence_scores": confidence_scores,
                "recommendation": recommendation
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def evaluate(self, test_images, test_labels):
        """
        Evaluate the classifier performance.
        
        Args:
            test_images (list): List of test images
            test_labels (list): List of corresponding labels
            
        Returns:
            dict: Performance metrics including accuracy, confusion matrix, etc.
        """
        if not self.is_trained:
            logger.warning("Model has not been trained yet")
            return {"success": False, "error": "Model has not been trained yet"}
        
        if len(test_images) == 0 or len(test_labels) == 0:
            logger.error("No test data provided")
            return {"success": False, "error": "No test data provided"}
        
        try:
            logger.info(f"Evaluating model on {len(test_images)} test images")
            
            # Extract features from test images
            features = []
            for image in test_images:
                image_features = extract_features(image)
                features.append(image_features)
            
            # Convert to numpy array
            features = np.array(features)
            
            # Generate predictions
            predictions = self.model.predict(features)
            
            # Calculate accuracy
            accuracy = accuracy_score(test_labels, predictions)
            
            # Generate confusion matrix
            cm = confusion_matrix(test_labels, predictions)
            
            # Generate classification report
            report = classification_report(test_labels, predictions, 
                                          target_names=self.ripeness_levels, 
                                          output_dict=True)
            
            logger.info(f"Model evaluation completed. Accuracy: {accuracy:.4f}")
            
            return {
                "success": True,
                "accuracy": accuracy,
                "confusion_matrix": cm,
                "classification_report": report
            }
            
        except Exception as e:
            logger.error(f"Error during model evaluation: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def save_model(self, file_path):
        """
        Save the trained model to disk.
        
        Args:
            file_path (str): Path to save the model
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_trained:
            logger.warning("Cannot save untrained model")
            return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save the model with pickle
            with open(file_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'ripeness_levels': self.ripeness_levels,
                    'is_trained': self.is_trained,
                    'feature_names': self.feature_names
                }, f)
            
            logger.info(f"Model saved to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model to {file_path}: {str(e)}")
            return False
    
    def load_model(self, file_path):
        """
        Load a trained model from disk.
        
        Args:
            file_path (str): Path to the saved model
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load the model with pickle
            with open(file_path, 'rb') as f:
                saved_data = pickle.load(f)
            
            self.model = saved_data['model']
            self.ripeness_levels = saved_data['ripeness_levels']
            self.is_trained = saved_data['is_trained']
            self.feature_names = saved_data.get('feature_names', [])
            
            logger.info(f"Model loaded from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model from {file_path}: {str(e)}")
            return False
    
    def _generate_recommendation(self, ripeness_level, confidence):
        """
        Generate a recommendation based on the ripeness level and confidence.
        
        Args:
            ripeness_level (str): Predicted ripeness level
            confidence (float): Confidence of the prediction
            
        Returns:
            dict: Recommendation details
        """
        # Initialize the recommendation
        recommendation = {
            "action": "",
            "description": "",
            "priority": 0,
            "confidence_note": ""
        }
        
        # Add confidence note if confidence is low
        if confidence < 0.7:
            recommendation["confidence_note"] = "Low confidence: Consider manual verification"
        
        # Generate recommendation based on ripeness level
        if ripeness_level == "unripe":
            recommendation["action"] = "Store"
            recommendation["description"] = "Keep in storage for ripening"
            recommendation["priority"] = 1
            
        elif ripeness_level == "ripe":
            recommendation["action"] = "Stock"
            recommendation["description"] = "Ideal for display and immediate sale"
            recommendation["priority"] = 2
            
        elif ripeness_level == "overripe":
            recommendation["action"] = "Discount"
            recommendation["description"] = "Discount for quick sale"
            recommendation["priority"] = 3
            
        elif ripeness_level == "spoiled":
            recommendation["action"] = "Discard"
            recommendation["description"] = "Remove from inventory"
            recommendation["priority"] = 4
        
        return recommendation
