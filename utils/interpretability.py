import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.inspection import permutation_importance

def get_feature_importance(model, X_test, feature_names):
    """
    Calculate feature importance for the model.
    
    Parameters:
    -----------
    model : dict
        Trained model and metadata
    X_test : numpy.ndarray
        Test features
    feature_names : list
        Names of features
    
    Returns:
    --------
    dict
        Dictionary with feature importance information
    """
    # Determine which model to use
    if model['fairness_applied'] and model['fair_model'] is not None:
        importance_model = model['fair_model']
    else:
        importance_model = model['original_model']
    
    # Try different methods to get feature importance
    if hasattr(importance_model, 'feature_importances_'):
        # For tree-based models
        importances = importance_model.feature_importances_
    else:
        # Use permutation importance as fallback
        result = permutation_importance(
            importance_model, X_test, np.zeros(X_test.shape[0]),
            n_repeats=10, random_state=42
        )
        importances = result.importances_mean
    
    # Ensure feature_names has the right length
    if len(feature_names) != len(importances):
        # Truncate or pad feature_names to match importances length
        if len(feature_names) > len(importances):
            feature_names = feature_names[:len(importances)]
        else:
            # Pad with generic names
            feature_names = list(feature_names) + [f'Feature_{i}' for i in range(len(feature_names), len(importances))]
    
    # Create a sorted dataframe of feature importances
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    return importance_df

def generate_explanation(model, X_instance):
    """
    Generate a human-readable explanation for a prediction.
    
    Parameters:
    -----------
    model : dict
        Trained model and metadata
    X_instance : pandas.DataFrame
        Single instance to explain
    
    Returns:
    --------
    tuple
        (explanation_text, shap_values)
    """
    # Determine which model to use
    if model['fairness_applied'] and model['fair_model'] is not None:
        explain_model = model['fair_model']
    else:
        explain_model = model['original_model']
    
    # Make prediction
    if hasattr(explain_model, 'predict_proba'):
        prediction = explain_model.predict(X_instance)[0]
        probability = explain_model.predict_proba(X_instance)[0, 1]
    else:
        prediction = explain_model.predict(X_instance)[0]
        probability = prediction  # Binary prediction as probability
    
    # Calculate SHAP values
    try:
        # Initialize the SHAP explainer based on model type
        if hasattr(explain_model, 'predict_proba'):
            explainer = shap.Explainer(explain_model)
        else:
            explainer = shap.Explainer(explain_model)
        
        # Calculate SHAP values
        shap_values = explainer(X_instance)
        
        # Get feature names and values
        feature_names = X_instance.columns
        feature_values = X_instance.values[0]
        
        # Sort features by absolute SHAP value
        if hasattr(shap_values, 'values'):
            # Newer SHAP versions
            feature_shap = list(zip(feature_names, feature_values, shap_values.values[0]))
        else:
            # Older SHAP versions
            feature_shap = list(zip(feature_names, feature_values, shap_values[0]))
        
        feature_shap.sort(key=lambda x: abs(x[2]), reverse=True)
        
        # Generate explanation text
        top_features = feature_shap[:5]  # Top 5 most important features
        
        explanation_parts = []
        
        # Prediction outcome
        if prediction == 1:
            explanation_parts.append(f"**Prediction: Student likely needs referral** (Probability: {probability:.1%})")
        else:
            explanation_parts.append(f"**Prediction: Student unlikely needs referral** (Probability: {probability:.1%})")
        
        explanation_parts.append("\n**Key factors influencing this prediction:**\n")
        
        # Add top factors
        for i, (name, value, shap_value) in enumerate(top_features):
            if shap_value > 0:
                direction = "increases"
                effect = "higher risk"
            else:
                direction = "decreases"
                effect = "lower risk"
            
            # Format the value based on type
            if isinstance(value, (int, float)):
                value_str = f"{value:.2f}" if isinstance(value, float) else str(value)
            else:
                value_str = str(value)
            
            explanation_parts.append(f"{i+1}. **{name}** = {value_str} {direction} the likelihood of referral (suggesting {effect})")
        
        # Overall explanation
        explanation = "\n".join(explanation_parts)
        
        return explanation, shap_values
    
    except Exception as e:
        # Fallback explanation if SHAP fails
        if prediction == 1:
            explanation = f"**Prediction: Student likely needs referral** (Probability: {probability:.1%})\n\n"
            explanation += "The model identified patterns in the student's data that suggest a referral may be beneficial."
        else:
            explanation = f"**Prediction: Student unlikely needs referral** (Probability: {probability:.1%})\n\n"
            explanation += "The model did not identify concerning patterns in the student's data that would suggest a referral is needed."
        
        return explanation, None

def plot_shap_values(shap_values):
    """
    Create a SHAP plot for a prediction.
    
    Parameters:
    -----------
    shap_values : shap.Explanation
        SHAP values for the instance
    
    Returns:
    --------
    matplotlib.figure.Figure
        SHAP plot
    """
    plt.figure(figsize=(10, 6))
    
    try:
        # Try to use SHAP's built-in plotting
        shap.plots.waterfall(shap_values[0], show=False)
        fig = plt.gcf()
        plt.tight_layout()
        return fig
    except:
        # Fallback to manual plotting
        try:
            # Get the feature values and names
            if hasattr(shap_values, 'data'):
                # Newer SHAP version
                feature_names = shap_values.feature_names
                feature_values = shap_values.data[0]
                shap_vals = shap_values.values[0]
            else:
                # Older SHAP version or different format
                feature_names = list(range(len(shap_values[0])))
                feature_values = None
                shap_vals = shap_values[0]
            
            # Sort features by absolute SHAP value
            if feature_values is not None:
                sorted_idx = np.argsort(np.abs(shap_vals))
                sorted_names = [feature_names[i] for i in sorted_idx]
                sorted_values = [feature_values[i] for i in sorted_idx]
                sorted_shap = shap_vals[sorted_idx]
            else:
                sorted_idx = np.argsort(np.abs(shap_vals))
                sorted_names = [feature_names[i] for i in sorted_idx]
                sorted_shap = shap_vals[sorted_idx]
            
            # Plot
            plt.barh(range(len(sorted_idx)), sorted_shap, color=['r' if x > 0 else 'b' for x in sorted_shap])
            plt.yticks(range(len(sorted_idx)), sorted_names)
            plt.xlabel('SHAP Value (Impact on Prediction)')
            plt.ylabel('Feature')
            plt.title('Feature Impact on Prediction')
            plt.tight_layout()
            
            return plt.gcf()
        
        except Exception as e:
            # If all else fails, return a blank figure with error message
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f"Unable to generate SHAP plot: {str(e)}", 
                     horizontalalignment='center', verticalalignment='center')
            plt.tight_layout()
            return plt.gcf()
