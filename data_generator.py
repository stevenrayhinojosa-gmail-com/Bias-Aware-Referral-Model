import numpy as np
import pandas as pd
import random

def generate_student_data(num_students=600):
    """
    Generate synthetic student data with racial bias in referrals.
    
    Parameters:
    -----------
    num_students : int
        Number of students to generate
    
    Returns:
    --------
    pandas.DataFrame
        Synthetic student data
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Define racial distribution based on provided demographics
    race_distribution = {
        'White': 0.31,              # 31% of students
        'African American': 0.13,   # 13% of students  
        'Hispanic': 0.31,           # 31% of students
        'Asian': 0.17,              # 17% of students
        'Two or More Races': 0.09,  # 9% of students
        'American Indian': 0.01,    # 1% of students
        'Pacific Islander': 0.01    # 1% of students
    }
    
    # Normalize the distribution to ensure it sums to 1
    total_percent = sum(race_distribution.values())
    race_distribution = {k: v/total_percent for k, v in race_distribution.items()}
    
    # Calculate number of students per race
    race_counts = {}
    assigned_students = 0
    
    for race, percent in race_distribution.items():
        if race == list(race_distribution.keys())[-1]:  # Last race gets remaining students
            race_counts[race] = num_students - assigned_students
        else:
            race_counts[race] = int(num_students * percent)
            assigned_students += race_counts[race]
    
    # Calculate exact referral numbers to achieve bias (African American students overrepresented)
    total_referrals = 120  # Fixed number of referrals (20% of 600 students)
    african_american_referrals = 40  # Disproportionate number for African American students
    
    # Student IDs
    student_ids = [f"S{i+1:04d}" for i in range(num_students)]
    
    # Race assignment
    races = []
    for race, count in race_counts.items():
        races.extend([race] * count)
    random.shuffle(races)  # Shuffle to avoid patterns
    
    # Create base DataFrame
    data = pd.DataFrame({
        'student_id': student_ids,
        'race': races
    })
    
    # Define feature parameters for each demographic group to simulate bias
    feature_params = {
        'African American': {
            'behavior_mean': 5.5, 'behavior_std': 1.5,
            'grades_mean': 70, 'grades_std': 15,
            'attendance_lambda': 2.0
        },
        'Hispanic': {
            'behavior_mean': 6.5, 'behavior_std': 1.5,
            'grades_mean': 75, 'grades_std': 15,
            'attendance_lambda': 1.5
        },
        'White': {
            'behavior_mean': 7.0, 'behavior_std': 1.5,
            'grades_mean': 80, 'grades_std': 15,
            'attendance_lambda': 1.0
        },
        'Asian': {
            'behavior_mean': 7.5, 'behavior_std': 1.2,
            'grades_mean': 85, 'grades_std': 12,
            'attendance_lambda': 0.8
        },
        'Two or More Races': {
            'behavior_mean': 6.8, 'behavior_std': 1.4,
            'grades_mean': 78, 'grades_std': 14,
            'attendance_lambda': 1.2
        },
        'American Indian': {
            'behavior_mean': 6.2, 'behavior_std': 1.6,
            'grades_mean': 72, 'grades_std': 16,
            'attendance_lambda': 1.8
        },
        'Pacific Islander': {
            'behavior_mean': 6.4, 'behavior_std': 1.5,
            'grades_mean': 74, 'grades_std': 15,
            'attendance_lambda': 1.6
        }
    }
    
    # Generate base feature distributions for each demographic group
    for race in race_distribution.keys():
        mask = data['race'] == race
        n_students = sum(mask)
        
        if n_students > 0:
            params = feature_params[race]
            
            # Behavior scores
            data.loc[mask, 'behavior_score'] = np.random.normal(
                params['behavior_mean'], params['behavior_std'], n_students
            ).clip(1, 10).round()
            
            # Grades
            data.loc[mask, 'grades'] = np.random.normal(
                params['grades_mean'], params['grades_std'], n_students
            ).clip(0, 100).round()
            
            # Attendance issues
            data.loc[mask, 'attendance_issues'] = np.random.poisson(
                params['attendance_lambda'], n_students
            ).clip(0, 5)
    
    # Initialize referrals with probabilities based on behavior, grades, and attendance
    data['referral_probability'] = (
        (10 - data['behavior_score']) * 0.05 +  # Lower behavior scores increase probability
        (100 - data['grades']) * 0.005 +        # Lower grades increase probability
        data['attendance_issues'] * 0.05         # More attendance issues increase probability
    )
    
    # Use the pre-calculated referral targets
    target_african_american_referrals = african_american_referrals  # Set to be disproportionate
    target_other_referrals = total_referrals - target_african_american_referrals
    
    # Ensure African American students get disproportionate referrals
    african_american_mask = data['race'] == 'African American'
    other_mask = ~african_american_mask
    
    # Sort African American students by current probability and mark top N for referral
    african_american_students = data[african_american_mask].copy()
    if len(african_american_students) > 0:
        african_american_students_sorted = african_american_students.sort_values('referral_probability', ascending=False)
        african_american_referral_count = min(target_african_american_referrals, len(african_american_students_sorted))
        african_american_students_sorted.iloc[:african_american_referral_count, african_american_students_sorted.columns.get_loc('referral_probability')] = 0.95
        african_american_students_sorted.iloc[african_american_referral_count:, african_american_students_sorted.columns.get_loc('referral_probability')] = 0.05
        data.loc[african_american_mask] = african_american_students_sorted
    else:
        african_american_referral_count = 0
        african_american_students_sorted = pd.DataFrame()
    
    # Sort other students and mark top N for referral
    other_students = data[other_mask].copy()
    if len(other_students) > 0:
        other_students_sorted = other_students.sort_values('referral_probability', ascending=False)
        other_referral_count = min(target_other_referrals, len(other_students_sorted))
        other_students_sorted.iloc[:other_referral_count, other_students_sorted.columns.get_loc('referral_probability')] = 0.95
        other_students_sorted.iloc[other_referral_count:, other_students_sorted.columns.get_loc('referral_probability')] = 0.05
        data.loc[other_mask] = other_students_sorted
    else:
        other_referral_count = 0
        other_students_sorted = pd.DataFrame()
    
    # Deterministic assignment of referrals based on probability thresholds
    # Static assignment to ensure exact control of the referral proportions
    data['referral'] = 0  # Initialize all to 0
    
    if len(african_american_students_sorted) > 0:
        data.loc[african_american_students_sorted.index[:african_american_referral_count], 'referral'] = 1
    
    if len(other_students_sorted) > 0:
        data.loc[other_students_sorted.index[:other_referral_count], 'referral'] = 1
    
    # Drop the temporary probability column
    data = data.drop('referral_probability', axis=1)
    
    # Convert to appropriate data types
    data['behavior_score'] = data['behavior_score'].astype(int)
    data['grades'] = data['grades'].astype(int)
    data['attendance_issues'] = data['attendance_issues'].astype(int)
    data['referral'] = data['referral'].astype(int)
    
    return data

if __name__ == "__main__":
    # Generate the data
    student_data = generate_student_data(600)
    
    # Print summary statistics to verify bias
    print("Data Summary:")
    print(f"Total students: {len(student_data)}")
    
    # Summarize data by race
    race_summary = student_data.groupby('race').agg({
        'student_id': 'count',
        'referral': 'sum'
    })
    race_summary = race_summary.rename(columns={'student_id': 'count'})
    race_summary['percentage_of_population'] = race_summary['count'] / len(student_data) * 100
    race_summary['percentage_of_referrals'] = race_summary['referral'] / race_summary['referral'].sum() * 100
    race_summary['referral_rate'] = race_summary['referral'] / race_summary['count'] * 100
    
    print("\nRace Statistics:")
    print(race_summary)
    
    # Save to CSV
    student_data.to_csv('student_data.csv', index=False)
    print("\nData saved to 'student_data.csv'")