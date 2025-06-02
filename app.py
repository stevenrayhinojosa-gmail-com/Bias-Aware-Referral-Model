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
from data_generator import load_referral_matrix

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

# Data Upload Section
st.header("Student Data Upload")

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

st.divider()

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
if 'model_feedback' not in st.session_state:
    st.session_state.model_feedback = {}
if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = False

if 'data_loaded' not in st.session_state:
    # Auto-load the generated dataset to show demographics and matrix
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
    
    # Load and display referral matrix analysis
    st.subheader("Educator Referral Decision Matrix")
    
    try:
        referral_matrix = load_referral_matrix()
        
        # Convert to DataFrame for analysis
        matrix_df = pd.DataFrame(referral_matrix)
        
        # Display the matrix
        st.write("**Behavioral Categories and Referral Decisions:**")
        
        # Group by category and show referral patterns
        category_analysis = matrix_df.groupby(['category', 'referred']).size().reset_index(name='count')
        category_pivot = category_analysis.pivot(index='category', columns='referred', values='count').fillna(0)
        category_pivot.columns = ['No Referral', 'Referral']
        category_pivot['Total Behaviors'] = category_pivot['No Referral'] + category_pivot['Referral']
        category_pivot['Referral Rate %'] = (category_pivot['Referral'] / category_pivot['Total Behaviors'] * 100).round(1)
        
        st.dataframe(category_pivot)
        
        # Show behavior severity levels
        st.write("**Behavior Severity Analysis:**")
        level_analysis = matrix_df.groupby(['level', 'referred']).size().reset_index(name='count')
        level_pivot = level_analysis.pivot(index='level', columns='referred', values='count').fillna(0)
        level_pivot.columns = ['No Referral', 'Referral']
        level_pivot['Total Behaviors'] = level_pivot['No Referral'] + level_pivot['Referral']
        level_pivot['Referral Rate %'] = (level_pivot['Referral'] / level_pivot['Total Behaviors'] * 100).round(1)
        
        st.dataframe(level_pivot)
        
        # Analyze student behavior patterns by race
        if 'primary_behavior' in st.session_state.data.columns:
            st.subheader("Student Behavior Patterns by Race")
            
            # Get behavior patterns by race
            behavior_by_race = st.session_state.data.groupby(['race', 'primary_behavior']).size().reset_index(name='count')
            
            # Filter out "No incident" for cleaner analysis
            behavior_incidents = behavior_by_race[behavior_by_race['primary_behavior'] != 'No incident']
            
            if len(behavior_incidents) > 0:
                # Create a pivot table showing behavior distribution by race
                behavior_pivot = behavior_incidents.pivot(index='primary_behavior', columns='race', values='count').fillna(0)
                
                # Calculate percentages within each race
                race_totals = st.session_state.data.groupby('race').size()
                behavior_percentages = behavior_pivot.div(race_totals, axis=1) * 100
                
                st.write("**Incident Rates by Race (% of students in each race):**")
                st.dataframe(behavior_percentages.round(2))
                
                # Identify potential bias contributors
                st.subheader("Bias Analysis Insights")
                
                # Find behaviors where African American students are overrepresented
                if 'African American' in behavior_percentages.columns:
                    african_american_rates = behavior_percentages['African American']
                    overall_avg = behavior_percentages.mean(axis=1)
                    
                    bias_indicators = african_american_rates - overall_avg
                    high_bias_behaviors = bias_indicators[bias_indicators > 2].sort_values(ascending=False)
                    
                    if len(high_bias_behaviors) > 0:
                        st.warning("**Behaviors contributing to potential bias:**")
                        for behavior, bias_level in high_bias_behaviors.items():
                            st.write(f"• **{behavior}**: African American students {bias_level:.1f}% above average")
                        
                        # Show category breakdown for high-bias behaviors
                        high_bias_behavior_names = high_bias_behaviors.index.tolist()
                        bias_behaviors_matrix = matrix_df[matrix_df['behavior'].isin(high_bias_behavior_names)]
                        
                        if len(bias_behaviors_matrix) > 0:
                            bias_categories = bias_behaviors_matrix['category'].value_counts()
                            st.write("**Categories most contributing to bias:**")
                            for category, count in bias_categories.items():
                                st.write(f"• **{category}**: {count} high-bias behaviors")
                    else:
                        st.success("No significant bias detected in behavior patterns")
                
                # Referral rate analysis by race
                referral_by_race = st.session_state.data.groupby('race')['referral'].agg(['sum', 'count']).reset_index()
                referral_by_race['referral_rate'] = (referral_by_race['sum'] / referral_by_race['count'] * 100).round(1)
                referral_by_race.columns = ['Race', 'Total Referrals', 'Total Students', 'Referral Rate %']
                
                st.write("**Referral Rates by Race:**")
                st.dataframe(referral_by_race)
                
                # Add bias analysis feedback buttons
                st.subheader("Bias Analysis Feedback")
                st.write("Was this bias analysis helpful and accurate?")
                
                col_bias_feedback1, col_bias_feedback2, col_bias_feedback3 = st.columns([1, 1, 2])
                
                with col_bias_feedback1:
                    if st.button("👍 Helpful Analysis", key="bias_thumbs_up"):
                        st.session_state.model_feedback['bias_analysis'] = {
                            'rating': 'positive',
                            'feedback_type': 'bias_analysis',
                            'timestamp': pd.Timestamp.now()
                        }
                        st.success("Thank you! Your feedback helps improve our bias detection.")
                
                with col_bias_feedback2:
                    if st.button("👎 Needs Improvement", key="bias_thumbs_down"):
                        st.session_state.model_feedback['bias_analysis'] = {
                            'rating': 'negative',
                            'feedback_type': 'bias_analysis',
                            'timestamp': pd.Timestamp.now()
                        }
                        st.error("Thank you for the feedback. We'll work to improve our bias analysis.")
                
                with col_bias_feedback3:
                    # Show current bias analysis feedback if exists
                    if 'bias_analysis' in st.session_state.model_feedback:
                        bias_feedback = st.session_state.model_feedback['bias_analysis']
                        if bias_feedback['rating'] == 'positive':
                            st.info("✓ You found this bias analysis helpful")
                        else:
                            st.info("✗ You rated this bias analysis as needing improvement")
        
    except FileNotFoundError:
        st.error("Referral matrix file not found. Please ensure the educator decision matrix is available.")
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
                        
                        # Add feedback buttons
                        st.subheader("Model Feedback")
                        st.write("Was this prediction helpful and accurate?")
                        
                        col_feedback1, col_feedback2, col_feedback3 = st.columns([1, 1, 2])
                        
                        with col_feedback1:
                            if st.button("👍 Good Prediction", key=f"thumbs_up_{selected_student_id}"):
                                st.session_state.model_feedback[selected_student_id] = {
                                    'rating': 'positive',
                                    'student_id': selected_student_id,
                                    'prediction': pred,
                                    'probability': prob,
                                    'timestamp': pd.Timestamp.now()
                                }
                                st.session_state.feedback_submitted = True
                                st.success("Thank you for your feedback!")
                        
                        with col_feedback2:
                            if st.button("👎 Poor Prediction", key=f"thumbs_down_{selected_student_id}"):
                                st.session_state.model_feedback[selected_student_id] = {
                                    'rating': 'negative',
                                    'student_id': selected_student_id,
                                    'prediction': pred,
                                    'probability': prob,
                                    'timestamp': pd.Timestamp.now()
                                }
                                st.session_state.feedback_submitted = True
                                st.error("Thank you for your feedback. We'll use this to improve the model.")
                        
                        with col_feedback3:
                            # Show current feedback if exists
                            if selected_student_id in st.session_state.model_feedback:
                                feedback = st.session_state.model_feedback[selected_student_id]
                                if feedback['rating'] == 'positive':
                                    st.info("✓ You rated this prediction as helpful")
                                else:
                                    st.info("✗ You rated this prediction as poor")
                    
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

# Feedback Summary Dashboard
if st.session_state.model_feedback:
    st.header("Model Performance Feedback")
    
    feedback_df = pd.DataFrame([
        {
            'Student ID': feedback['student_id'],
            'Rating': feedback['rating'],
            'Prediction': 'Referral' if feedback['prediction'] else 'No Referral',
            'Confidence': f"{feedback['probability']:.1%}",
            'Timestamp': feedback['timestamp'].strftime('%Y-%m-%d %H:%M')
        }
        for feedback in st.session_state.model_feedback.values()
    ])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Feedback Summary")
        
        # Calculate feedback statistics
        total_feedback = len(feedback_df)
        positive_feedback = len(feedback_df[feedback_df['Rating'] == 'positive'])
        negative_feedback = len(feedback_df[feedback_df['Rating'] == 'negative'])
        
        # Display metrics
        col_metric1, col_metric2, col_metric3 = st.columns(3)
        
        with col_metric1:
            st.metric("Total Feedback", total_feedback)
        
        with col_metric2:
            satisfaction_rate = (positive_feedback / total_feedback * 100) if total_feedback > 0 else 0
            st.metric("Satisfaction Rate", f"{satisfaction_rate:.1f}%")
        
        with col_metric3:
            st.metric("Positive Ratings", positive_feedback)
        
        # Feedback breakdown chart
        if total_feedback > 0:
            fig, ax = plt.subplots(figsize=(6, 4))
            labels = ['Positive', 'Negative']
            sizes = [positive_feedback, negative_feedback]
            colors = ['#5cb85c', '#d9534f']
            
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.set_title('User Feedback Distribution')
            plt.tight_layout()
            st.pyplot(fig)
    
    with col2:
        st.subheader("Recent Feedback")
        
        # Display recent feedback table
        if len(feedback_df) > 0:
            # Sort by timestamp (most recent first)
            feedback_display = feedback_df.sort_values('Timestamp', ascending=False)
            
            # Add rating icons
            feedback_display['Rating'] = feedback_display['Rating'].map({
                'positive': '👍 Good',
                'negative': '👎 Poor'
            })
            
            st.dataframe(feedback_display, use_container_width=True)
            
            # Feedback insights
            st.subheader("Insights")
            
            if negative_feedback > 0:
                # Analyze patterns in negative feedback
                negative_predictions = feedback_df[feedback_df['Rating'] == 'negative']['Prediction'].value_counts()
                
                st.write("**Areas for improvement:**")
                for pred_type, count in negative_predictions.items():
                    percentage = (count / negative_feedback * 100)
                    st.write(f"• {count} poor ratings for {pred_type} predictions ({percentage:.1f}%)")
            
            if positive_feedback >= total_feedback * 0.8:
                st.success("High user satisfaction! The model is performing well.")
            elif positive_feedback >= total_feedback * 0.6:
                st.info("Good user satisfaction. Some areas for improvement identified.")
            else:
                st.warning("User satisfaction could be improved. Consider model refinement.")
        
        # Option to export feedback data
        if st.button("Export Feedback Data"):
            csv = feedback_df.to_csv(index=False)
            st.download_button(
                label="Download Feedback CSV",
                data=csv,
                file_name=f"model_feedback_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
# If no data is available, show basic information
if st.session_state.data is None:
    st.header("About This System")
    
    st.markdown("""
    ### How it works
    
    This system analyzes student referral patterns to identify potential bias in educational decision-making.
    
    ### Key Features
    
    - **Bias Detection**: Identifies disparities in referral rates across demographic groups
    - **Behavioral Analysis**: Shows which behavioral categories contribute most to bias
    - **Educator Decision Matrix**: Displays real referral decision patterns
    - **Transparency**: Provides clear analysis of referral patterns and potential bias sources
    
    ### Ethical Principles
    
    This system is designed with these principles in mind:
    
    - **Fairness**: Identifying and addressing bias across demographic groups
    - **Transparency**: Providing clear analysis of referral decision patterns
    - **Support-oriented**: Focusing on improving equity in educational support systems
    """)
