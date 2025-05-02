# Ethical Student Referral Prediction System

An ethical machine learning system for predicting student referrals with fairness constraints and interpretability features. This application helps educators make more equitable decisions about student referrals while mitigating racial bias.

## Purpose

Many schools have documented disparities in referral rates across different racial groups. This application demonstrates how fairness-aware machine learning can help educators:

1. Analyze existing bias in referral data
2. Compare predictions with and without fairness constraints
3. Make more equitable referral decisions
4. Understand the factors that influence predictions

## Features

- **Bias Analysis Dashboard**: Visualizes racial disparities in the existing referral data
- **Fairness Constraints**: Allows selecting fairness metrics like demographic parity and equal opportunity
- **Model Comparison**: Train both fair and unfair models to see the impact of fairness constraints
- **Interactive Visualization**: Compare referral rates across demographic groups
- **Individual Student Analysis**: Get detailed explanations for predictions for specific students
- **Interpretability**: Understand what factors influence the model's predictions

## Data Requirements

The system works with student data in CSV format with the following columns:
- `student_id`: Unique identifier for each student 
- `race`: Demographic information (e.g., "Black", "White", "Hispanic")
- `behavior_score`: Numeric score (1-10)
- `grades`: Numeric values (0-100)
- `attendance_issues`: Numeric values (0-5)
- `referral`: Binary values (0 or 1) indicating whether a student received a referral

## Getting Started

1. Launch the application
2. Select data source (use the included sample dataset or upload your own)
3. Set fairness constraints in the sidebar
4. Train both the fair and unfair models
5. Explore the racial bias analysis, fairness comparison, and prediction results
6. Select individual students to analyze prediction explanations

## Sample Dataset

The included `student_data.csv` demonstrates a synthetic dataset with racial bias in referrals:
- Black students represent 15% of the population but receive 33% of all referrals
- This allows users to see how fairness constraints can help mitigate this bias

## Ethical Principles

This system is designed according to these ethical principles:
- **Fairness**: Ensuring predictions are free from bias across demographic groups
- **Transparency**: Providing clear explanations for model decisions
- **Privacy**: Protecting sensitive student information
- **Support-oriented**: Focusing on identifying students who need additional resources, not punitive measures

## Implementation Details

The system uses:
- Streamlit for the interactive web interface
- Scikit-learn for machine learning models
- Fairness-aware algorithms to mitigate bias
- Interpretability methods to explain predictions

## Use Cases

1. **School Administrators**: Identify and address systemic bias in disciplinary actions
2. **Teachers**: Get fair and explainable recommendations for student referrals
3. **Diversity & Inclusion Staff**: Monitor and improve equity in disciplinary processes
4. **Educational Researchers**: Study the impact of bias mitigation on educational outcomes

## Limitations

- The fairness constraints may not address all forms of bias present in the data
- The explanations are approximations of complex model behavior
- Real-world deployment would require additional privacy and security safeguards

## License

Open source - freely available for educational and non-commercial purposes.