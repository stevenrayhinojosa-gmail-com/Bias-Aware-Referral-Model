import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_model(X_train, y_train, sensitive_train, fairness_constraints=None):
    """
    Train a machine learning model with fairness constraints.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training features
    y_train : pandas.Series
        Training target
    sensitive_train : pandas.Series
        Sensitive attributes for training data
    fairness_constraints : dict
        Dictionary of fairness constraints to apply
        
    Returns:
    --------
    dict
        Trained model and metadata
    """
    # Default fairness constraints if none provided
    if fairness_constraints is None:
        fairness_constraints = {
            'demographic_parity': True,
            'equal_opportunity': True,
            'threshold': 0.7
        }

    # Initialize model
    base_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight='balanced'  # Use balanced class weights to help with fairness
    )
    
    # Train the model
    base_model.fit(X_train, y_train)
    
    # Get predictions and probabilities for original model
    y_pred_orig = base_model.predict(X_train)
    y_prob_orig = base_model.predict_proba(X_train)[:, 1]
    
    # Implement reweighting for fairness if constraints are enabled
    if fairness_constraints['demographic_parity'] or fairness_constraints['equal_opportunity']:
        # Convert sensitive_train to numpy array if it's a pandas Series
        if isinstance(sensitive_train, pd.Series):
            sensitive_train = sensitive_train.values
        
        # Get unique sensitive feature values
        sensitive_values = np.unique(sensitive_train)
        
        # Calculate metrics for each sensitive group
        group_metrics = {}
        for value in sensitive_values:
            mask = (sensitive_train == value)
            
            # Calculate group-specific prediction rates
            group_metrics[value] = {
                'size': np.sum(mask),
                'prediction_rate': np.mean(y_pred_orig[mask]),
                'true_positive_rate': np.sum((y_pred_orig == 1) & (y_train.values == 1) & mask) / 
                                     np.sum((y_train.values == 1) & mask) if np.sum((y_train.values == 1) & mask) > 0 else 0,
                'false_positive_rate': np.sum((y_pred_orig == 1) & (y_train.values == 0) & mask) / 
                                      np.sum((y_train.values == 0) & mask) if np.sum((y_train.values == 0) & mask) > 0 else 0
            }
        
        # Find maximum and minimum rates
        max_pred_rate = max(metrics['prediction_rate'] for metrics in group_metrics.values())
        min_pred_rate = min(metrics['prediction_rate'] for metrics in group_metrics.values())
        
        max_tpr = max(metrics['true_positive_rate'] for metrics in group_metrics.values())
        min_tpr = min(metrics['true_positive_rate'] for metrics in group_metrics.values())
        
        # Check if fairness constraints are violated
        demographic_parity_violated = (max_pred_rate - min_pred_rate) > (1 - fairness_constraints['threshold'])
        equal_opportunity_violated = (max_tpr - min_tpr) > (1 - fairness_constraints['threshold'])
        
        # Apply fairness mitigation if needed
        if (fairness_constraints['demographic_parity'] and demographic_parity_violated) or \
           (fairness_constraints['equal_opportunity'] and equal_opportunity_violated):
            
            # Create sample weights based on sensitive features to mitigate bias
            sample_weights = np.ones(len(y_train))
            
            for value in sensitive_values:
                mask = (sensitive_train == value)
                
                # Adjust weights for demographic parity
                if fairness_constraints['demographic_parity'] and demographic_parity_violated:
                    weight_factor = 1.0 / group_metrics[value]['prediction_rate'] if group_metrics[value]['prediction_rate'] > 0 else 1.0
                    # Normalize weight factor
                    weight_factor = weight_factor / max(1.0 / metrics['prediction_rate'] 
                                                       for metrics in group_metrics.values() 
                                                       if metrics['prediction_rate'] > 0)
                    sample_weights[mask] *= weight_factor
                
                # Adjust weights for equal opportunity
                if fairness_constraints['equal_opportunity'] and equal_opportunity_violated:
                    # Only adjust weights for positive cases
                    pos_mask = mask & (y_train.values == 1)
                    if np.any(pos_mask):
                        tpr_weight_factor = 1.0 / group_metrics[value]['true_positive_rate'] if group_metrics[value]['true_positive_rate'] > 0 else 1.0
                        # Normalize weight factor
                        tpr_weight_factor = tpr_weight_factor / max(1.0 / metrics['true_positive_rate'] 
                                                                   for metrics in group_metrics.values() 
                                                                   if metrics['true_positive_rate'] > 0)
                        sample_weights[pos_mask] *= tpr_weight_factor
            
            # Retrain model with sample weights
            fair_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            )
            
            fair_model.fit(X_train, y_train, sample_weight=sample_weights)
            
            # Store both models
            model = {
                'original_model': base_model,
                'fair_model': fair_model,
                'fairness_applied': True
            }
        else:
            # No fairness mitigation needed
            model = {
                'original_model': base_model,
                'fair_model': None,
                'fairness_applied': False
            }
    else:
        # No fairness constraints enabled
        model = {
            'original_model': base_model,
            'fair_model': None,
            'fairness_applied': False
        }
    
    return model

def predict_referrals(model, data):
    """
    Make predictions using the trained model.
    
    Parameters:
    -----------
    model : dict
        Trained model and metadata
    data : pandas.DataFrame
        Data to make predictions on
    
    Returns:
    --------
    dict
        Prediction results
    """
    # Preprocess the data for prediction (exclude target)
    from utils.data_processing import preprocess_data
    
    # Try to identify the features by excluding what's definitely not a feature
    exclude_cols = ['student_id', 'race']
    
    # Check if referral column exists in the data
    referral_cols = [col for col in data.columns if 'referral' in col.lower()]
    if referral_cols:
        exclude_cols.extend(referral_cols)
    
    # Extract features
    feature_cols = [col for col in data.columns if col not in exclude_cols]
    X = data[feature_cols]
    
    # Determine which model to use for prediction
    if model['fairness_applied'] and model['fair_model'] is not None:
        pred_model = model['fair_model']
    else:
        pred_model = model['original_model']
    
    # Check if preprocessing is needed
    if hasattr(pred_model, 'feature_names_in_'):
        # Model was trained on preprocessed data, so we need to preprocess
        X_transformed, _, _, _ = preprocess_data(data)
        
        # Make predictions
        predictions = pred_model.predict(X_transformed)
        probabilities = pred_model.predict_proba(X_transformed)[:, 1]
    else:
        # Model can handle raw features
        predictions = pred_model.predict(X)
        probabilities = pred_model.predict_proba(X)[:, 1]
    
    return {
        'prediction': predictions,
        'probability': probabilities
    }
