import streamlit as st
import pandas as pd
import numpy as np
# Temporarily comment out plotly to proceed without it
# import plotly.express as px
import matplotlib.pyplot as plt
import os
import io

from utils.data_processing import preprocess_data, split_data, validate_data
from utils.model import train_model, predict_referrals
from utils.fairness import calculate_fairness_metrics, mitigate_bias
from utils.interpretability import get_feature_importance, generate_explanation, plot_shap_values

# Set page configuration
st.set_page_config(
    page_title="Ethical Referral Prediction System",
    page_icon="📚",
    layout="wide",
)

# Application title and description
st.title("Ethical Referral Prediction System")
st.markdown("""
This system helps teachers predict student referrals while ensuring fairness and providing 
interpretable explanations for predictions. Our goal is to support educational decision-making 
while preventing racial bias and promoting equity.
""")

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'fairness_metrics' not in st.session_state:
    st.session_state.fairness_metrics = None
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'training_completed' not in st.session_state:
    st.session_state.training_completed = False
if 'selected_student' not in st.session_state:
    st.session_state.selected_student = None
if 'explanation' not in st.session_state:
    st.session_state.explanation = None
if 'shap_values' not in st.session_state:
    st.session_state.shap_values = None
if 'fairness_constraints' not in st.session_state:
    st.session_state.fairness_constraints = {
        'demographic_parity': True,
        'equal_opportunity': True,
        'threshold': 0.7
    }
if 'data_loaded' not in st.session_state:
    # Try to auto-load the dataset on first run
    try:
        data = pd.read_csv('student_data.csv')
        validation_result, validation_message = validate_data(data)
        if validation_result:
            st.session_state.data = data
            st.session_state.data_loaded = True
        else:
            st.session_state.data = None
            st.session_state.data_loaded = False
    except Exception:
        st.session_state.data = None
        st.session_state.data_loaded = False

# Sidebar for application controls
with st.sidebar:
    st.header("Controls")

    # Data Selection Section
    st.subheader("1. Student Data")
    
    data_option = st.radio(
        "Choose data source:",
        ["Use sample dataset", "Upload my own data"]
    )
    
    if data_option == "Use sample dataset":
        try:
            # Load the pre-generated dataset
            data = pd.read_csv('student_data.csv')
            
            # Validate data
            validation_result, validation_message = validate_data(data)
            
            if validation_result:
                st.session_state.data = data
                st.success("Sample dataset loaded successfully!")
            else:
                st.error(f"Invalid data format in sample dataset: {validation_message}")
                st.session_state.data = None
        except Exception as e:
            st.error(f"Error reading sample dataset: {e}")
            st.session_state.data = None
    else:
        # Data Upload Section
        uploaded_file = st.file_uploader("Upload CSV file with student data", type=["csv"])
        
        if uploaded_file is not None:
            try:
                # Read data
                data = pd.read_csv(uploaded_file)
                
                # Validate data
                validation_result, validation_message = validate_data(data)
                
                if validation_result:
                    st.session_state.data = data
                    st.success("Data uploaded successfully!")
                else:
                    st.error(f"Invalid data format: {validation_message}")
                    st.session_state.data = None
            except Exception as e:
                st.error(f"Error reading file: {e}")
                st.session_state.data = None
    
    # Fairness Constraints Section
    st.subheader("2. Fairness Constraints")
    st.session_state.fairness_constraints['demographic_parity'] = st.checkbox(
        "Ensure Demographic Parity", 
        value=st.session_state.fairness_constraints['demographic_parity'],
        help="Ensures similar referral rates across protected groups"
    )
    
    st.session_state.fairness_constraints['equal_opportunity'] = st.checkbox(
        "Ensure Equal Opportunity",
        value=st.session_state.fairness_constraints['equal_opportunity'],
        help="Ensures similar true positive rates across protected groups"
    )
    
    st.session_state.fairness_constraints['threshold'] = st.slider(
        "Fairness Threshold", 
        min_value=0.5, 
        max_value=1.0, 
        value=st.session_state.fairness_constraints['threshold'],
        step=0.05,
        help="Higher values enforce stricter fairness"
    )
    
    # Model Training Section
    st.subheader("3. Model Training")
    
    train_button = st.button("Train Model", disabled=st.session_state.data is None)
    
    if train_button and st.session_state.data is not None:
        with st.spinner("Training model with fairness constraints..."):
            # Preprocess data
            X, y, sensitive_features, feature_names = preprocess_data(st.session_state.data)
            
            # Split data
            X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = split_data(
                X, y, sensitive_features
            )
            
            # Train model with fairness constraints
            st.session_state.model = train_model(
                X_train, y_train, sensitive_train, 
                fairness_constraints=st.session_state.fairness_constraints
            )
            
            # Calculate fairness metrics
            st.session_state.fairness_metrics = calculate_fairness_metrics(
                st.session_state.model, X_test, y_test, sensitive_test
            )
            
            # Calculate feature importance
            st.session_state.feature_importance = get_feature_importance(
                st.session_state.model, X_test, feature_names
            )
            
            # Generate predictions
            st.session_state.predictions = predict_referrals(
                st.session_state.model, st.session_state.data
            )
            
            # Update state
            st.session_state.training_completed = True
            st.session_state.shap_values = None
            st.success("Model trained successfully!")
    
    # Reset application
    if st.button("Reset Application"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

# Main content area
if st.session_state.data is not None:
    # Show data overview
    st.header("Data Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Student Data Sample")
        st.dataframe(st.session_state.data.head(5))
    
    with col2:
        st.subheader("Data Statistics")
        st.write(f"Number of students: {len(st.session_state.data)}")
        
        if 'race' in st.session_state.data.columns:
            # Create a simple matplotlib pie chart instead of plotly
            race_counts = st.session_state.data['race'].value_counts()
            fig, ax = plt.subplots()
            ax.pie(race_counts, labels=race_counts.index, autopct='%1.1f%%')
            ax.set_title('Demographic Distribution')
            st.pyplot(fig)
    
    # Add a section for bias analysis
    st.header("Racial Bias Analysis")
    if 'race' in st.session_state.data.columns and 'referral' in st.session_state.data.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            # Referrals by race analysis
            race_referral_data = st.session_state.data.groupby('race').agg({
                'referral': ['count', 'sum']
            })
            
            # Flatten multi-index columns
            race_referral_data.columns = ['_'.join(col).strip() for col in race_referral_data.columns.values]
            
            # Calculate percentage of referrals
            race_referral_data['percent_of_group_referred'] = (race_referral_data['referral_sum'] / race_referral_data['referral_count'] * 100).round(1)
            race_referral_data['percent_of_total_referrals'] = (race_referral_data['referral_sum'] / race_referral_data['referral_sum'].sum() * 100).round(1)
            race_referral_data['percent_of_population'] = (race_referral_data['referral_count'] / race_referral_data['referral_count'].sum() * 100).round(1)
            race_referral_data['disparity_index'] = (race_referral_data['percent_of_total_referrals'] / race_referral_data['percent_of_population']).round(2)
            
            st.subheader("Referral Statistics by Race")
            st.dataframe(race_referral_data[['percent_of_population', 'percent_of_group_referred', 'percent_of_total_referrals', 'disparity_index']])
            
            # Add interpretation
            black_disparity = race_referral_data.loc['Black', 'disparity_index'] if 'Black' in race_referral_data.index else 0
            if black_disparity > 1.5:
                st.error(f"⚠️ Black students are {black_disparity}x overrepresented in referrals")
            elif black_disparity > 1.1:
                st.warning(f"⚠️ Black students are slightly overrepresented in referrals (disparity index: {black_disparity})")
        
        with col2:
            # Create a bar chart comparing population percentage to referral percentage
            races = race_referral_data.index
            pop_percent = race_referral_data['percent_of_population']
            ref_percent = race_referral_data['percent_of_total_referrals']
            
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(races))
            width = 0.35
            
            bar1 = ax.bar(x - width/2, pop_percent, width, label='% of Population')
            bar2 = ax.bar(x + width/2, ref_percent, width, label='% of Referrals')
            
            ax.set_xlabel('Race')
            ax.set_ylabel('Percentage')
            ax.set_title('Population vs. Referral Percentage by Race')
            ax.set_xticks(x)
            ax.set_xticklabels(races)
            ax.legend()
            
            # Add value labels on top of bars
            for bar in [bar1, bar2]:
                for rect in bar:
                    height = rect.get_height()
                    ax.annotate(f'{height}%',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig)
    
    # Show model results if training is completed
    if st.session_state.training_completed:
        st.header("Model Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Fairness Metrics")
            
            metrics_df = pd.DataFrame({
                'Metric': list(st.session_state.fairness_metrics.keys()),
                'Value': list(st.session_state.fairness_metrics.values())
            })
            
            st.dataframe(metrics_df, use_container_width=True)
            
            # Add fairness interpretation
            demo_parity = st.session_state.fairness_metrics.get('demographic_parity_difference', 1.0)
            eq_opportunity = st.session_state.fairness_metrics.get('equal_opportunity_difference', 1.0)
            
            if demo_parity < 0.1 and eq_opportunity < 0.1:
                st.success("✅ Model predictions are fairly balanced across demographic groups")
            elif demo_parity < 0.2 and eq_opportunity < 0.2:
                st.warning("⚠️ Model shows moderate fairness concerns")
            else:
                st.error("❌ Model shows significant fairness issues that need addressing")
        
        with col2:
            st.subheader("Feature Importance")
            
            # Plot feature importance using matplotlib instead of plotly
            if st.session_state.feature_importance is not None:
                fig, ax = plt.subplots(figsize=(10, 6))
                importance = st.session_state.feature_importance['importance']
                features = st.session_state.feature_importance['feature']
                # Sort by importance
                sort_idx = importance.argsort()
                ax.barh(features[sort_idx], importance[sort_idx])
                ax.set_title('Feature Importance')
                ax.set_xlabel('Importance')
                ax.set_ylabel('Feature')
                plt.tight_layout()
                st.pyplot(fig)
        
        # Prediction results
        st.header("Prediction Results")
        
        if st.session_state.predictions is not None:
            # Display predictions
            results_df = st.session_state.data.copy()
            results_df['referral_probability'] = st.session_state.predictions['probability']
            results_df['predicted_referral'] = st.session_state.predictions['prediction']
            
            st.dataframe(
                results_df[['student_id', 'referral_probability', 'predicted_referral'] + 
                          [col for col in results_df.columns if col not in 
                           ['student_id', 'referral_probability', 'predicted_referral']]],
                use_container_width=True
            )
            
            # Individual student analysis
            st.header("Individual Student Analysis")
            
            # Student selection
            student_ids = results_df['student_id'].astype(str).tolist()
            selected_student_id = st.selectbox(
                "Select a student for detailed analysis:",
                student_ids
            )
            
            if selected_student_id:
                student_index = results_df[results_df['student_id'].astype(str) == selected_student_id].index[0]
                st.session_state.selected_student = student_index
                
                student_data = results_df.iloc[student_index]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Student Information")
                    
                    # Display relevant student information
                    info_to_display = {k: v for k, v in student_data.items() 
                                     if k not in ['referral_probability', 'predicted_referral']}
                    
                    for key, value in info_to_display.items():
                        st.write(f"**{key}:** {value}")
                
                with col2:
                    st.subheader("Prediction")
                    
                    prob = student_data['referral_probability']
                    pred = student_data['predicted_referral']
                    
                    # Create simple donut chart with matplotlib
                    fig, ax = plt.subplots(figsize=(8, 8))
                    sizes = [prob, 1-prob]
                    labels = ['Referral Risk', 'Low Risk']
                    colors = ['#FF5555', '#AAAAAA']
                    
                    # Create a donut chart
                    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                          wedgeprops=dict(width=0.5))
                    
                    # Add text in center
                    ax.text(0, 0, f"{prob:.1%}", ha='center', va='center', fontsize=20)
                    
                    # Equal aspect ratio ensures that pie is drawn as a circle
                    ax.set_aspect('equal')
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    
                    if pred:
                        st.error("⚠️ Student predicted to need referral")
                    else:
                        st.success("✅ Student predicted not to need referral")
                
                # Generate explanation for the prediction
                if st.button("Generate Explanation for This Prediction"):
                    with st.spinner("Generating explanation..."):
                        # Get SHAP values for explanation
                        X_instance = results_df.iloc[student_index:student_index+1].drop(
                            ['student_id', 'referral_probability', 'predicted_referral'], axis=1
                        )
                        
                        explanation, shap_values = generate_explanation(
                            st.session_state.model, X_instance
                        )
                        
                        st.session_state.explanation = explanation
                        st.session_state.shap_values = shap_values
                
                # Display explanation if available
                if st.session_state.explanation is not None:
                    st.subheader("Prediction Explanation")
                    
                    st.markdown(st.session_state.explanation)
                    
                    # Display SHAP plot
                    if st.session_state.shap_values is not None:
                        st.subheader("Factor Influence (SHAP values)")
                        
                        fig = plot_shap_values(st.session_state.shap_values)
                        st.pyplot(fig)
else:
    # Display instructions when no data is uploaded
    st.info("👈 Please upload student data using the sidebar to begin.")
    
    # Display information about the system
    st.header("About This System")
    
    st.markdown("""
    ### How it works
    
    1. **Upload student data**: Provide a CSV file with student information, including academic, behavioral, and demographic data.
    
    2. **Set fairness constraints**: Choose which fairness metrics to enforce to ensure the model treats all demographic groups fairly.
    
    3. **Train the model**: The system builds a predictive model that adheres to ethical AI principles and the selected fairness constraints.
    
    4. **Review predictions**: Examine the model's predictions and fairness metrics to ensure they meet educational and ethical standards.
    
    5. **Analyze individual students**: Get detailed explanations for why specific students may need referrals or support.
    
    ### Required Data Format
    
    Your CSV file should include:
    
    - `student_id`: Unique identifier for each student
    - Academic features (e.g., grades, attendance)
    - Behavioral metrics (e.g., past incidents, classroom behavior)
    - `race`: Demographic information as a protected attribute
    - Other relevant information for making referral predictions
    
    ### Ethical Principles
    
    This system is designed with these principles in mind:
    
    - **Fairness**: Ensuring predictions are free from bias across demographic groups
    - **Transparency**: Providing clear explanations for model decisions
    - **Privacy**: Protecting sensitive student information
    - **Support-oriented**: Focusing on identifying students who need additional resources, not punitive measures
    """)
