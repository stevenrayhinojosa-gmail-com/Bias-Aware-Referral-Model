import numpy as np
import pandas as pd
import random

def generate_student_data(num_students=300):
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
    
    # Calculate number of students per race to meet the 15% Black requirement
    num_black = int(num_students * 0.15)  # 15% Black
    num_hispanic = int(num_students * 0.25)  # 25% Hispanic
    num_white = num_students - num_black - num_hispanic  # 60% White
    
    # Student IDs
    student_ids = [f"S{i+1:03d}" for i in range(num_students)]
    
    # Race assignment
    races = (["Black"] * num_black + 
             ["Hispanic"] * num_hispanic + 
             ["White"] * num_white)
    random.shuffle(races)  # Shuffle to avoid patterns
    
    # Create base DataFrame
    data = pd.DataFrame({
        'student_id': student_ids,
        'race': races
    })
    
    # Generate base feature distributions - different for each race to simulate bias
    for race in ['Black', 'Hispanic', 'White']:
        mask = data['race'] == race
        n_students = sum(mask)
        
        # Behavior scores - slightly lower for Black students to simulate bias
        if race == 'Black':
            data.loc[mask, 'behavior_score'] = np.random.normal(5.5, 1.5, n_students).clip(1, 10).round()
        elif race == 'Hispanic':
            data.loc[mask, 'behavior_score'] = np.random.normal(6.5, 1.5, n_students).clip(1, 10).round()
        else: # White
            data.loc[mask, 'behavior_score'] = np.random.normal(7.0, 1.5, n_students).clip(1, 10).round()
        
        # Grades - slightly lower for Black students to simulate bias
        if race == 'Black':
            data.loc[mask, 'grades'] = np.random.normal(70, 15, n_students).clip(0, 100).round()
        elif race == 'Hispanic':
            data.loc[mask, 'grades'] = np.random.normal(75, 15, n_students).clip(0, 100).round()
        else: # White
            data.loc[mask, 'grades'] = np.random.normal(80, 15, n_students).clip(0, 100).round()
        
        # Attendance issues - slightly higher for Black students to simulate bias
        if race == 'Black':
            data.loc[mask, 'attendance_issues'] = np.random.poisson(2.0, n_students).clip(0, 5)
        elif race == 'Hispanic':
            data.loc[mask, 'attendance_issues'] = np.random.poisson(1.5, n_students).clip(0, 5)
        else: # White
            data.loc[mask, 'attendance_issues'] = np.random.poisson(1.0, n_students).clip(0, 5)
    
    # Initialize referrals with probabilities based on behavior, grades, and attendance
    data['referral_probability'] = (
        (10 - data['behavior_score']) * 0.05 +  # Lower behavior scores increase probability
        (100 - data['grades']) * 0.005 +        # Lower grades increase probability
        data['attendance_issues'] * 0.05         # More attendance issues increase probability
    )
    
    # Adjust referral probabilities to ensure race bias
    # Black students should be 15% of population but 33% of referrals
    num_referrals = int(num_students * 0.20)     # Overall referral rate (e.g., 20% of all students)
    target_black_referrals = int(num_referrals * 0.33)  # 33% of all referrals should be Black students
    target_non_black_referrals = num_referrals - target_black_referrals
    
    # Ensure black students get disproportionate referrals
    black_mask = data['race'] == 'Black'
    non_black_mask = ~black_mask
    
    # Sort Black students by current probability and mark top N for referral
    black_students = data[black_mask].copy()
    black_students_sorted = black_students.sort_values('referral_probability', ascending=False)
    black_referral_count = min(target_black_referrals, len(black_students_sorted))
    black_students_sorted.iloc[:black_referral_count, black_students_sorted.columns.get_loc('referral_probability')] = 0.95
    black_students_sorted.iloc[black_referral_count:, black_students_sorted.columns.get_loc('referral_probability')] = 0.05
    data.loc[black_mask] = black_students_sorted
    
    # Sort non-Black students and mark top N for referral
    non_black_students = data[non_black_mask].copy()
    non_black_students_sorted = non_black_students.sort_values('referral_probability', ascending=False)
    non_black_referral_count = min(target_non_black_referrals, len(non_black_students_sorted))
    non_black_students_sorted.iloc[:non_black_referral_count, non_black_students_sorted.columns.get_loc('referral_probability')] = 0.95
    non_black_students_sorted.iloc[non_black_referral_count:, non_black_students_sorted.columns.get_loc('referral_probability')] = 0.05
    data.loc[non_black_mask] = non_black_students_sorted
    
    # Deterministic assignment of referrals based on probability thresholds
    # Static assignment to ensure exact control of the referral proportions
    black_mask = data['race'] == 'Black'
    data.loc[black_students_sorted.index[:black_referral_count], 'referral'] = 1
    data.loc[black_students_sorted.index[black_referral_count:], 'referral'] = 0
    
    data.loc[non_black_students_sorted.index[:non_black_referral_count], 'referral'] = 1
    data.loc[non_black_students_sorted.index[non_black_referral_count:], 'referral'] = 0
    
    # Double check that we have the right % of Black referrals (~33%)
    black_referral_pct = data[data['race'] == 'Black']['referral'].sum() / data['referral'].sum()
    
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
    student_data = generate_student_data(300)
    
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