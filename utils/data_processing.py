import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

def validate_data(data):
    """
    Validate that the uploaded data has the required format and features.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The uploaded data to validate
    
    Returns:
    --------
    tuple
        (bool, str) indicating if validation passed and any error message
    """
    # Check if data is empty
    if data is None or data.empty:
        return False, "Data is empty"
    
    # Check for required columns
    required_columns = ['student_id']
    
    if not all(col in data.columns for col in required_columns):
        missing = [col for col in required_columns if col not in data.columns]
        return False, f"Missing required columns: {', '.join(missing)}"
    
    # Check for protected attribute (for fairness metrics)
    if 'race' not in data.columns:
        return False, "Missing 'race' column needed for fairness metrics"
    
    # Check for sufficient features to make predictions
    if len(data.columns) < 5:  # At least some features plus required ones
        return False, "Insufficient features for prediction (need at least 5 columns)"
    
    # Check for duplicate student IDs
    if data['student_id'].duplicated().any():
        return False, "Duplicate student IDs found"
    
    return True, "Data validation passed"

def preprocess_data(data):
    """
    Preprocess the data for model training.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The data to preprocess
    
    Returns:
    --------
    tuple
        (X, y, sensitive_features, feature_names)
    """
    # Determine target variable - assuming 'referral' is the target
    # If not present, we'll try to infer from column names or create a mock
    if 'referral' in data.columns:
        target_col = 'referral'
    elif 'has_referral' in data.columns:
        target_col = 'has_referral'
    elif 'referred' in data.columns:
        target_col = 'referred'
    else:
        # Check if any column has 'referral' in its name
        referral_cols = [col for col in data.columns if 'referral' in col.lower()]
        if referral_cols:
            target_col = referral_cols[0]
        else:
            raise ValueError("No referral column found in the data. Please ensure your data has a target column.")
    
    # Identify numeric and categorical features
    # Exclude ID, target, and sensitive attributes
    exclude_cols = ['student_id', target_col, 'race']
    feature_cols = [col for col in data.columns if col not in exclude_cols]
    
    numeric_features = data[feature_cols].select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = data[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Create preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Extract features, target, and sensitive attribute
    X = data[feature_cols]
    y = data[target_col].astype(int)
    sensitive_features = data['race']
    
    # Fit and transform the data
    X_transformed = preprocessor.fit_transform(X)
    
    # Get feature names after one-hot encoding
    feature_names = []
    
    # Add numeric feature names
    feature_names.extend(numeric_features)
    
    # Add one-hot encoded feature names
    if categorical_features:
        ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
        categorical_names = ohe.get_feature_names_out(categorical_features)
        feature_names.extend(categorical_names)
    
    return X_transformed, y, sensitive_features, feature_names

def split_data(X, y, sensitive_features, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    y : pandas.Series
        Target variable
    sensitive_features : pandas.Series
        Sensitive attributes
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test, sensitive_train, sensitive_test)
    """
    X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
        X, y, sensitive_features, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, sensitive_train, sensitive_test
