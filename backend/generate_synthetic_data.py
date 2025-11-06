import pandas as pd
import numpy as np
import random

# Load the existing courses to use as a base for generation
courses_df = pd.read_csv('./data/courses.csv')

# Constants
NUM_STUDENTS_PER_COURSE = 500
GRADES = ['A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D', 'D-', 'E']
SUBJECTS = ['Mathematics', 'Kiswahili', 'English', 'Arabic', 'German', 'French', 'Chemistry', 'Physics', 'Biology', 'Home Science', 'Agriculture', 'Computer Studies', 'History', 'Geography', 'Religious Education', 'Life Skills', 'Business Studies', 'Music', 'Art and Design', 'Drawing and Design', 'Building Construction', 'Power and Mechanics', 'Metalwork', 'Aviation', 'Woodwork', 'Electronics']
SKILLS_POOL = ['Problem Solving', 'Logical Thinking', 'Communication', 'Empathy', 'Research', 'Leadership', 'Analytical Skills', 'Manual Dexterity', 'Creativity', 'Teamwork']
INTERESTS_POOL = ['Programming', 'Healthcare', 'Debating', 'Management', 'Building things', 'Technology', 'Science', 'Art', 'Music', 'Sports']

all_student_data = []

print(f"Generating synthetic data for {len(courses_df)} courses...")

for _, course in courses_df.iterrows():
    course_id = course['course_id']
    required_subjects = [s.strip() for s in course['required_subjects'].split(',')]
    required_skills = [s.strip() for s in course['required_skills'].split(',')]

    for _ in range(NUM_STUDENTS_PER_COURSE):
        student = {}
        
        # Generate grades
        student_grades = {}
        for sub in SUBJECTS:
            if sub in required_subjects:
                # Higher probability of good grades for required subjects
                grade = np.random.choice(GRADES, p=[0.2, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.0, 0.0])
            else:
                # Random grades for other subjects
                grade = np.random.choice(GRADES)
            student_grades[sub] = grade
        
        student.update(student_grades)

        # Generate interests and skills
        num_interests = random.randint(1, 3)
        student_interests = random.sample(INTERESTS_POOL, num_interests)
        # Ensure some required skills are present
        student_interests.extend(random.sample(required_skills, k=min(len(required_skills), 1)))
        student['interests'] = ', '.join(list(set(student_interests)))

        num_skills = random.randint(1, 4)
        student_skills = random.sample(SKILLS_POOL, num_skills)
        student_skills.extend(random.sample(required_skills, k=min(len(required_skills), 2)))
        student['skills'] = ', '.join(list(set(student_skills)))

        # The target variable
        student['target_course_id'] = course_id

        all_student_data.append(student)

# Create the final DataFrame
synthetic_df = pd.DataFrame(all_student_data)

# Save to CSV
output_path = './data/synthetic_student_data.csv'
synthetic_df.to_csv(output_path, index=False)

print(f"Successfully generated and saved {len(synthetic_df)} synthetic student records to {output_path}")
