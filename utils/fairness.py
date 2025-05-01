import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def calculate_fairness_metrics(model, X_test, y_test, sensitive_test):
    """
    Calculate fairness metrics for the model.
    
    Parameters:
    -----------
    model : dict
        Trained model and metadata
    X_test : numpy.ndarray
        Test features
    y_test : pandas.Series
        Test target
    sensitive_test : pandas.Series
        Sensitive attributes for test data
    
    Returns:
    --------
    dict
        Dictionary of fairness metrics
    """
    # Determine which model to use for evaluation
    if model['fairness_applied'] and model['fair_model'] is not None:
        eval_model = model['fair_model']
    else:
        eval_model = model['original_model']
    
    # Make predictions
    y_pred = eval_model.predict(X_test)
    y_prob = eval_model.predict_proba(X_test)[:, 1]
    
    # Convert to numpy arrays if they aren't already
    if isinstance(y_test, pd.Series):
        y_test = y_test.values
    if isinstance(sensitive_test, pd.Series):
        sensitive_test = sensitive_test.values
    
    # Get unique sensitive feature values
    sensitive_values = np.unique(sensitive_test)
    
    # Calculate overall performance metrics
    overall_metrics = {
        'accuracy': np.mean(y_pred == y_test),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    # Calculate group-specific metrics
    group_metrics = {}
    for value in sensitive_values:
        mask = (sensitive_test == value)
        
        if np.sum(mask) > 0:
            # Calculate confusion matrix for this group
            tn, fp, fn, tp = confusion_matrix(y_test[mask], y_pred[mask], labels=[0, 1]).ravel()
            
            # Calculate basic rates
            group_metrics[value] = {
                'size': np.sum(mask),
                'base_rate': np.mean(y_test[mask]),
                'prediction_rate': np.mean(y_pred[mask]),
                'true_positive_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'accuracy': (tp + tn) / (tp + tn + fp + fn)
            }
    
    # Calculate fairness metrics
    fairness_metrics = {}
    
    # 1. Demographic Parity (also known as Statistical Parity)
    # Measures if prediction rates are the same across protected groups
    prediction_rates = [metrics['prediction_rate'] for metrics in group_metrics.values()]
    fairness_metrics['demographic_parity_difference'] = max(prediction_rates) - min(prediction_rates)
    fairness_metrics['demographic_parity_ratio'] = min(prediction_rates) / max(prediction_rates) if max(prediction_rates) > 0 else 1.0
    
    # 2. Equal Opportunity
    # Measures if true positive rates are the same across protected groups
    tpr_values = [metrics['true_positive_rate'] for metrics in group_metrics.values()]
    fairness_metrics['equal_opportunity_difference'] = max(tpr_values) - min(tpr_values)
    fairness_metrics['equal_opportunity_ratio'] = min(tpr_values) / max(tpr_values) if max(tpr_values) > 0 else 1.0
    
    # 3. Equalized Odds
    # Measures if both TPR and FPR are the same across protected groups
    fpr_values = [metrics['false_positive_rate'] for metrics in group_metrics.values()]
    tpr_disparity = max(tpr_values) - min(tpr_values)
    fpr_disparity = max(fpr_values) - min(fpr_values)
    fairness_metrics['equalized_odds_difference'] = max(tpr_disparity, fpr_disparity)
    
    # 4. Accuracy Parity
    # Measures if accuracy is the same across protected groups
    accuracy_values = [metrics['accuracy'] for metrics in group_metrics.values()]
    fairness_metrics['accuracy_parity_difference'] = max(accuracy_values) - min(accuracy_values)
    
    # Add overall performance metrics
    fairness_metrics.update({f'overall_{k}': v for k, v in overall_metrics.items()})
    
    # Add group metrics for detailed analysis
    fairness_metrics['group_metrics'] = group_metrics
    
    return fairness_metrics

def mitigate_bias(model, X, y, sensitive_features, fairness_constraints):
    """
    Apply bias mitigation techniques to improve fairness.
    
    Parameters:
    -----------
    model : object
        Original trained model
    X : numpy.ndarray
        Features
    y : pandas.Series
        Target variable
    sensitive_features : pandas.Series
        Sensitive attributes
    fairness_constraints : dict
        Dictionary of fairness constraints to apply
    
    Returns:
    --------
    object
        Debiased model
    """
    # Convert to numpy arrays if they aren't already
    if isinstance(y, pd.Series):
        y = y.values
    if isinstance(sensitive_features, pd.Series):
        sensitive_features = sensitive_features.values
    
    # Get unique sensitive feature values
    sensitive_values = np.unique(sensitive_features)
    
    # Get predictions from the original model
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        y_prob = model.predict(X).astype(float)
    
    # Calculate group-specific metrics
    group_metrics = {}
    for value in sensitive_values:
        mask = (sensitive_features == value)
        
        # Calculate basic rates
        group_metrics[value] = {
            'size': np.sum(mask),
            'base_rate': np.mean(y[mask]),
            'prediction_rate': np.mean(y_prob[mask] >= 0.5),
            'mean_prediction': np.mean(y_prob[mask])
        }
    
    # Apply threshold adjustments for each group based on fairness constraints
    threshold_adjustments = {}
    
    if fairness_constraints.get('demographic_parity', False):
        # Find the group with the highest prediction rate as reference
        reference_rate = max(metrics['prediction_rate'] for metrics in group_metrics.values())
        
        # Calculate adjustments to match the reference rate
        for value in sensitive_values:
            current_rate = group_metrics[value]['prediction_rate']
            current_mean = group_metrics[value]['mean_prediction']
            
            if current_rate < reference_rate:
                # Lower threshold to increase positive predictions
                adjustment_factor = 0.9  # Start with a small adjustment
                threshold_adjustments[value] = 0.5 * adjustment_factor
            else:
                threshold_adjustments[value] = 0.5  # No adjustment needed
    
    # Return the original model and threshold adjustments
    return {
        'original_model': model,
        'threshold_adjustments': threshold_adjustments
    }

def precision_score(y_true, y_pred):
    """Calculate precision score safely"""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def recall_score(y_true, y_pred):
    """Calculate recall score safely"""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def f1_score(y_true, y_pred):
    """Calculate F1 score safely"""
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
