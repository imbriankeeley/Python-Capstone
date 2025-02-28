"""
Recommendation generation logic for the Fruit Ripeness Classification System.

File path: /utils/recommendation.py
"""

def generate_recommendation(ripeness_level, confidence_score):
    """
    Generate a recommendation based on ripeness level and confidence score.
    
    Args:
        ripeness_level (str): Predicted ripeness level
        confidence_score (float): Confidence of the prediction
        
    Returns:
        dict: Recommendation details including:
            - action: Recommended action
            - description: Detailed description
            - confidence_modifier: Any modifier based on confidence
            - priority: Priority level (1-4, where 4 is highest priority)
    """
    # Initialize recommendation dictionary
    recommendation = {
        "action": "",
        "description": "",
        "priority": 0,
        "confidence_note": ""
    }
    
    # Add confidence note if confidence is low
    if confidence_score < 0.7:
        recommendation["confidence_note"] = "Low confidence prediction: Consider manual verification"
    
    # Generate recommendation based on ripeness level
    if ripeness_level == "unripe":
        recommendation["action"] = "Store"
        recommendation["description"] = "Keep in storage for ripening (3-5 days)"
        recommendation["priority"] = 1
        
    elif ripeness_level == "ripe":
        recommendation["action"] = "Stock"
        recommendation["description"] = "Ideal for immediate display and sale"
        recommendation["priority"] = 2
        
    elif ripeness_level == "overripe":
        recommendation["action"] = "Discount"
        recommendation["description"] = "Mark for quick sale at reduced price"
        recommendation["priority"] = 3
        
    elif ripeness_level == "spoiled":
        recommendation["action"] = "Discard"
        recommendation["description"] = "Remove from inventory immediately"
        recommendation["priority"] = 4
    
    # Further adjust recommendation for very low confidence
    if confidence_score < 0.5:
        recommendation = adjust_recommendation_for_confidence(recommendation, confidence_score)
    
    return recommendation

def get_action_priority(ripeness_level):
    """
    Get the priority level for the recommended action.
    
    Args:
        ripeness_level (str): Predicted ripeness level
        
    Returns:
        int: Priority level (1-4, where 4 is highest priority)
    """
    priorities = {
        "unripe": 1,   # Lowest priority - normal storage
        "ripe": 2,     # Medium priority - normal sale
        "overripe": 3, # High priority - needs attention soon
        "spoiled": 4   # Highest priority - immediate action required
    }
    
    return priorities.get(ripeness_level.lower(), 1)  # Default to lowest priority if unknown

def get_display_text(recommendation):
    """
    Get user-friendly display text for a recommendation.
    
    Args:
        recommendation (dict): Recommendation details
        
    Returns:
        str: Formatted display text
    """
    # Get basic components
    action = recommendation.get("action", "")
    description = recommendation.get("description", "")
    confidence_note = recommendation.get("confidence_note", "")
    
    # Format display text
    if confidence_note:
        return f"{action}: {description}\n\nNote: {confidence_note}"
    else:
        return f"{action}: {description}"

def adjust_recommendation_for_confidence(recommendation, confidence_score):
    """
    Adjust recommendation based on confidence score.
    
    Args:
        recommendation (dict): Original recommendation
        confidence_score (float): Confidence of the prediction
        
    Returns:
        dict: Adjusted recommendation
    """
    # Create a copy of the recommendation to avoid modifying the original
    adjusted = recommendation.copy()
    
    # For very low confidence, suggest manual verification regardless of ripeness level
    if confidence_score < 0.4:
        adjusted["confidence_note"] = "Very low confidence: Manual inspection required"
        
        # For potentially high-priority items (overripe/spoiled), adjust the action
        if recommendation["priority"] >= 3:
            adjusted["action"] = "Check"
            adjusted["description"] = "Possible quality issue. Manually verify before taking action."
            # Keep the priority high to ensure attention
        
        # For lower priority items, suggest rechecking
        else:
            adjusted["description"] += " (Manual verification recommended)"
    
    return adjusted
