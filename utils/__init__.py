"""
Utility functions package for the Fruit Ripeness Classification System.

This package contains utility functions for image processing,
recommendation generation, and visualization used throughout the application.

File path: /utils/__init__.py
"""

from utils.image_processing import (
    preprocess_image, 
    normalize_image, 
    segment_fruit,
    apply_image_transforms,
    extract_image_for_display
)

from utils.recommendation import (
    generate_recommendation,
    get_action_priority,
    get_display_text,
    adjust_recommendation_for_confidence
)

from utils.visualization_utils import (
    create_confusion_matrix_figure,
    create_feature_heatmap_figure,
    create_distribution_plot_figure,
    create_ripeness_comparison_figure,
    create_feature_importance_bar_chart,
    create_combined_results_figure,
    save_figure
)

__all__ = [
    # Image processing
    'preprocess_image',
    'normalize_image',
    'segment_fruit',
    'apply_image_transforms',
    'extract_image_for_display',
    
    # Recommendation
    'generate_recommendation',
    'get_action_priority',
    'get_display_text',
    'adjust_recommendation_for_confidence',
    
    # Visualization
    'create_confusion_matrix_figure',
    'create_feature_heatmap_figure',
    'create_distribution_plot_figure',
    'create_ripeness_comparison_figure',
    'create_feature_importance_bar_chart',
    'create_combined_results_figure',
    'save_figure'
]
