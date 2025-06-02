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
    page_title="Bias Aware Referral Prediction System",
    page_icon="📚",
    layout="wide",
)

# Application title and description
st.title("Bias Aware Referral Prediction System")
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
# Store results from both fair and unfair models for comparison
if 'fair_model_results' not in st.session_state:
    st.session_state.fair_model_results = None
if 'unfair_model_results' not in st.session_state:
    st.session_state.unfair_model_results = None

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
    
    col1, col2 = st.columns(2)
    
    with col1:
        train_button = st.button("Train Bias Aware Model", disabled=st.session_state.data is None)
    
    with col2:
        train_unfair_button = st.button("Train Standard Model", disabled=st.session_state.data is None)
    
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
            
            # Save fair model results
            st.session_state.fair_model_results = {
                'model': st.session_state.model,
                'fairness_metrics': st.session_state.fairness_metrics,
                'feature_importance': st.session_state.feature_importance,
                'predictions': st.session_state.predictions
            }
            
            # Update state
            st.session_state.training_completed = True
            st.session_state.model_type = "fair"
            st.session_state.shap_values = None
            st.success("Bias Aware model trained successfully!")
    
    if train_unfair_button and st.session_state.data is not None:
        with st.spinner("Training model without fairness constraints..."):
            # Preprocess data
            X, y, sensitive_features, feature_names = preprocess_data(st.session_state.data)
            
            # Split data
            X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = split_data(
                X, y, sensitive_features
            )
            
            # Train model WITHOUT fairness constraints
            unfair_constraints = {
                'demographic_parity': False,
                'equal_opportunity': False,
                'threshold': 0.5
            }
            
            st.session_state.model = train_model(
                X_train, y_train, sensitive_train, 
                fairness_constraints=unfair_constraints
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
            
            # Save unfair model results
            st.session_state.unfair_model_results = {
                'model': st.session_state.model,
                'fairness_metrics': st.session_state.fairness_metrics,
                'feature_importance': st.session_state.feature_importance,
                'predictions': st.session_state.predictions
            }
            
            # Update state
            st.session_state.training_completed = True
            st.session_state.model_type = "unfair"
            st.session_state.shap_values = None
            st.warning("Model trained WITHOUT fairness constraints!")
    
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
            # Create box plots showing distribution of key metrics by race
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # Grades by race
            race_groups = [st.session_state.data[st.session_state.data['race'] == race]['grades'].values 
                          for race in st.session_state.data['race'].unique()]
            axes[0, 0].boxplot(race_groups, labels=st.session_state.data['race'].unique())
            axes[0, 0].set_title('Grades by Race')
            axes[0, 0].set_ylabel('Grades')
            
            # Behavior score by race
            race_groups = [st.session_state.data[st.session_state.data['race'] == race]['behavior_score'].values 
                          for race in st.session_state.data['race'].unique()]
            axes[0, 1].boxplot(race_groups, labels=st.session_state.data['race'].unique())
            axes[0, 1].set_title('Behavior Score by Race')
            axes[0, 1].set_ylabel('Behavior Score')
            
            # Attendance issues by race
            race_groups = [st.session_state.data[st.session_state.data['race'] == race]['attendance_issues'].values 
                          for race in st.session_state.data['race'].unique()]
            axes[1, 0].boxplot(race_groups, labels=st.session_state.data['race'].unique())
            axes[1, 0].set_title('Attendance Issues by Race')
            axes[1, 0].set_ylabel('Attendance Issues')
            
            # Student count by race (bar chart)
            race_counts = st.session_state.data['race'].value_counts()
            axes[1, 1].bar(race_counts.index, race_counts.values)
            axes[1, 1].set_title('Student Count by Race')
            axes[1, 1].set_ylabel('Number of Students')
            
            # Add percentage labels on the bar chart
            total_students = len(st.session_state.data)
            for i, (race, count) in enumerate(race_counts.items()):
                percentage = (count / total_students) * 100
                axes[1, 1].text(i, count + 0.5, f'{percentage:.1f}%', ha='center', va='bottom')
            
            plt.tight_layout()
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
            # Show model type
            if 'model_type' in st.session_state and st.session_state.model_type == "fair":
                st.subheader("Fairness Metrics (Bias Aware Model)")
                st.success("This model was trained with bias awareness constraints")
            elif 'model_type' in st.session_state and st.session_state.model_type == "unfair":
                st.subheader("Fairness Metrics (Standard Model)")
                st.warning("This model was trained WITHOUT bias awareness")
            else:
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
        
        # Fairness Comparison Section (if both models have been trained)
        if st.session_state.fair_model_results is not None and st.session_state.unfair_model_results is not None:
            st.header("Fairness Comparison")
            st.info("This section compares predictions from models with and without fairness constraints")
            
            # Create DataFrame for comparison
            fair_predictions = st.session_state.fair_model_results['predictions']
            unfair_predictions = st.session_state.unfair_model_results['predictions']
            
            # Calculate referral rates by race
            race_fair_refs = {}
            race_unfair_refs = {}
            
            for race in st.session_state.data['race'].unique():
                # Get race indices
                race_indices = st.session_state.data[st.session_state.data['race'] == race].index
                
                # Calculate fair model referral rate for this race
                race_fair_refs[race] = fair_predictions['prediction'][race_indices].mean() * 100
                
                # Calculate unfair model referral rate for this race
                race_unfair_refs[race] = unfair_predictions['prediction'][race_indices].mean() * 100
            
            # Create plot comparing referral rates by race between models
            fig, ax = plt.subplots(figsize=(10, 6))
            races = list(race_fair_refs.keys())
            x = np.arange(len(races))
            width = 0.35
            
            fair_rates = [race_fair_refs[race] for race in races]
            unfair_rates = [race_unfair_refs[race] for race in races]
            
            bar1 = ax.bar(x - width/2, fair_rates, width, label='Bias Aware Model', color='#5cb85c')
            bar2 = ax.bar(x + width/2, unfair_rates, width, label='Standard Model', color='#d9534f')
            
            ax.set_xlabel('Race')
            ax.set_ylabel('Referral Rate (%)')
            ax.set_title('Comparison of Referral Rates by Race Between Models')
            ax.set_xticks(x)
            ax.set_xticklabels(races)
            ax.legend()
            
            # Add value labels on top of bars
            for bar in [bar1, bar2]:
                for rect in bar:
                    height = rect.get_height()
                    ax.annotate(f'{height:.1f}%',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Calculate overall differences between models
            fair_refs_count = fair_predictions['prediction'].sum()
            unfair_refs_count = unfair_predictions['prediction'].sum()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Impact on Students")
                st.write(f"Bias Aware Model: {fair_refs_count} total referrals ({fair_refs_count/len(st.session_state.data)*100:.1f}%)")
                st.write(f"Standard Model: {unfair_refs_count} total referrals ({unfair_refs_count/len(st.session_state.data)*100:.1f}%)")
                
                if 'Black' in race_fair_refs and 'White' in race_fair_refs:
                    # Calculate disparity reduction
                    unfair_disparity = race_unfair_refs['Black'] / race_unfair_refs['White']
                    fair_disparity = race_fair_refs['Black'] / race_fair_refs['White']
                    improvement = (unfair_disparity - fair_disparity) / unfair_disparity * 100
                    
                    st.write(f"Racial disparity reduction: {improvement:.1f}%")
                    
                    if improvement > 30:
                        st.success("✅ Significant reduction in racial disparities")
                    elif improvement > 10:
                        st.info("ℹ️ Moderate reduction in racial disparities")
                    else:
                        st.warning("⚠️ Limited impact on racial disparities")
            
            with col2:
                # Identify students with different predictions between models
                diff_predictions = []
                fair_pred = fair_predictions['prediction']
                unfair_pred = unfair_predictions['prediction']
                
                for i in range(len(fair_pred)):
                    if fair_pred[i] != unfair_pred[i]:
                        diff_predictions.append(i)
                
                st.subheader("Prediction Changes")
                st.write(f"{len(diff_predictions)} students ({len(diff_predictions)/len(st.session_state.data)*100:.1f}%) have different predictions between models")
                
                # Count by race
                race_changes = {}
                for i in diff_predictions:
                    race = st.session_state.data.iloc[i]['race']
                    if race not in race_changes:
                        race_changes[race] = 0
                    race_changes[race] += 1
                
                # Create a summary table
                if race_changes:
                    changes_df = pd.DataFrame({
                        'Race': list(race_changes.keys()),
                        'Students Affected': list(race_changes.values()),
                        'Percent of Race': [race_changes[race] / (st.session_state.data['race'] == race).sum() * 100 
                                           for race in race_changes.keys()]
                    })
                    
                    st.dataframe(changes_df)
                
            # Add a note about impact
            st.info("""
            ### Understanding the Impact
            
            The comparison above shows how applying fairness constraints affects referral predictions across different racial groups.
            
            - A standard model without fairness constraints may amplify existing biases in the data.
            - The fair model adjusts predictions to ensure similar referral rates across protected groups.
            - Some students receive different referral predictions between the models, which highlights the impact of bias mitigation.
            """)
        
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
                    
                    # Current model prediction
                    prob = student_data['referral_probability']
                    pred = student_data['predicted_referral']
                    
                    # Create two tabs for current prediction and comparison
                    current_tab, comparison_tab = st.tabs(["Current Model", "Fair vs. Unfair"])
                    
                    with current_tab:
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
                    
                    with comparison_tab:
                        # Only show comparison if both models have been trained
                        if st.session_state.fair_model_results is not None and st.session_state.unfair_model_results is not None:
                            fair_pred = st.session_state.fair_model_results['predictions']
                            unfair_pred = st.session_state.unfair_model_results['predictions']
                            
                            # Get probabilities
                            fair_prob = fair_pred['probability'][student_index]
                            unfair_prob = unfair_pred['probability'][student_index]
                            fair_ref = fair_pred['prediction'][student_index]
                            unfair_ref = unfair_pred['prediction'][student_index]
                            
                            # Create a comparison chart
                            fig, ax = plt.subplots(figsize=(10, 5))
                            x = [0, 1]
                            y = [unfair_prob, fair_prob]
                            
                            ax.bar(x, y, width=0.6, color=['#d9534f', '#5cb85c'])
                            ax.set_xticks(x)
                            ax.set_xticklabels(['Standard Model', 'Bias Aware Model'])
                            ax.set_ylabel('Referral Probability')
                            ax.set_title('Model Comparison for This Student')
                            ax.set_ylim(0, 1)
                            
                            # Add value labels on top of bars
                            for i, v in enumerate(y):
                                ax.annotate(f"{v:.1%}",
                                            xy=(i, v),
                                            xytext=(0, 3),
                                            textcoords="offset points",
                                            ha='center', va='bottom')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Show predictions
                            col1, col2 = st.columns(2)
                            with col1:
                                if unfair_ref:
                                    st.error("⚠️ Standard model: Referral")
                                else:
                                    st.success("✅ Standard model: No referral")
                                
                            with col2:
                                if fair_ref:
                                    st.error("⚠️ Bias Aware model: Referral")
                                else:
                                    st.success("✅ Bias Aware model: No referral")
                            
                            # Add interpretation if predictions differ
                            if fair_ref != unfair_ref:
                                st.warning("❓ Different predictions between models!")
                                if fair_ref and not unfair_ref:
                                    st.info("The Bias Aware model flagged this student for referral while the standard model did not, possibly to balance referral rates across demographic groups.")
                                else:
                                    st.info("The Bias Aware model did not flag this student for referral while the standard model did, possibly to reduce overrepresentation of their demographic group.")
                        else:
                            st.info("Train both fair and unfair models to see comparison for this student.")
                
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
