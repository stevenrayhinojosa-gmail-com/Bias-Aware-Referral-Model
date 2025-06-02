import numpy as np
import pandas as pd
import random
import csv

def load_referral_matrix(filename='referral_matrix_no_mixed.csv'):
    """Load the referral decision matrix from CSV file."""
    referral_matrix = []
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            referral_matrix.append({
                'level': int(row['level']),
                'category': row['category'],
                'behavior': row['behavior'],
                'referred': row['referred'] == 'Yes',
                'recorded': row['recorded'] == 'Yes'
            })
    return referral_matrix

def generate_student_behaviors(student_data, referral_matrix, bias_multiplier=None):
    """
    Generate realistic student behaviors and referral decisions based on educator matrix.
    
    Parameters:
    -----------
    student_data : pandas.DataFrame
        Basic student information
    referral_matrix : list
        Behavioral decision matrix from educators
    bias_multiplier : dict
        Optional bias multipliers by race for referral likelihood
    
    Returns:
    --------
    pandas.DataFrame
        Enhanced student data with behaviors and referral decisions
    """
    if bias_multiplier is None:
        # Default bias - African American students more likely to be referred
        bias_multiplier = {
            'African American': 2.5,
            'Hispanic': 1.3,
            'Pacific Islander': 1.2,
            'American Indian': 1.1,
            'Two or More Races': 0.9,
            'White': 0.6,
            'Asian': 0.4
        }
    
    # Get behaviors that lead to referrals vs those that don't
    referral_behaviors = [b for b in referral_matrix if b['referred']]
    non_referral_behaviors = [b for b in referral_matrix if not b['referred']]
    
    behaviors = []
    referrals = []
    
    for idx, row in student_data.iterrows():
        race = row['race']
        
        # 30% of students have some behavioral incident
        if random.random() < 0.30:
            # Apply racial bias in behavior selection
            race_bias = bias_multiplier.get(race, 1.0)
            
            # Higher bias means more likely to get a referral-worthy behavior assigned
            if random.random() < (0.15 * race_bias):  # Base 15% chance of referral behavior
                # Select a behavior that leads to referral
                behavior = random.choice(referral_behaviors)
                behaviors.append(behavior['behavior'])
                referrals.append(1)
            else:
                # Select a behavior that doesn't lead to referral
                behavior = random.choice(non_referral_behaviors)
                behaviors.append(behavior['behavior'])
                referrals.append(0)
        else:
            # No behavioral incident
            behaviors.append('No incident')
            referrals.append(0)
    
    student_data['primary_behavior'] = behaviors
    student_data['referral'] = referrals
    
    return student_data

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
    
    # Load the referral matrix for realistic behavior-based decisions
    referral_matrix = load_referral_matrix()
    
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
    
    # Use the referral matrix to generate realistic behavior-based referrals
    data = generate_student_behaviors(data, referral_matrix)
    
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