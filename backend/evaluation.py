# This is a placeholder script for evaluating the recommender system.
# To be meaningful, this script requires a labeled "ground truth" dataset.
# This dataset would contain student profiles (grades, interests, skills) and the 
# "correct" or ideal course/career recommendation for that profile.

import pandas as pd
from app.recommender import Recommender
from app.models import Student


def evaluate_recommender():
    """
    This function demonstrates how to evaluate the recommender system.
    It uses a small, sample ground truth dataset to calculate accuracy and precision.

    - Accuracy: (Number of correct recommendations) / (Total number of recommendations)
    - Precision@K: For each user, is the correct item in the top K recommendations?
    """
    
    print("Starting recommender system evaluation...")

    # --- 1. Sample Ground Truth Data ---
    # In a real-world scenario, this would be a large CSV file.
    ground_truth_data = [
        {
            "student": {"grades": {"Mathematics": "A", "Physics": "A-", "English": "B"}, "interests": ["programming", "gaming"], "skills": ["logical thinking"]},
            "correct_course": "Bachelor of Science in Computer Science",
            "correct_career": "Software Engineer"
        },
        {
            "student": {"grades": {"Biology": "A", "Chemistry": "A", "English": "A-"}, "interests": ["helping people", "science"], "skills": ["empathy", "communication"]},
            "correct_course": "Bachelor of Medicine and Bachelor of Surgery",
            "correct_career": "Doctor"
        },
        {
            "student": {"grades": {"History": "A", "English": "A", "Kiswahili": "A"}, "interests": ["debating", "reading"], "skills": ["research", "critical thinking"]},
            "correct_course": "Bachelor of Laws",
            "correct_career": "Lawyer"
        },
        {
            "student": {"grades": {"Mathematics": "C", "Physics": "D", "Agriculture": "A"}, "interests": ["farming", "business"], "skills": ["manual dexterity"]},
            "correct_course": "Diploma in Electrical Engineering", # Mismatch example
            "correct_career": "Electrician"
        }
    ]

    # --- 2. Initialize the Recommender ---
    recommender = Recommender(
        courses_path="./data/courses.csv",
        careers_path="./data/careers.csv",
        model_path="./app/course_recommender_model.joblib",
        metrics_path="./app/model_metrics.json"
    )

    # --- 3. Perform Evaluation ---
    correct_course_predictions = 0
    correct_career_predictions = 0
    k = 1 # We are checking if the top-1 recommendation is correct

    for item in ground_truth_data:
        student_profile = Student(**item["student"])
        recommendations = recommender.recommend(student_profile)
        
        # Get the top-1 recommendation
        top_course_rec = recommendations['courses'][0]['name']
        top_career_rec = recommendations['careers'][0]['name']

        print(f"\nStudent Profile: {student_profile.grades}, {student_profile.interests}")
        print(f"  - Top Course Rec: {top_course_rec} (Correct: {item['correct_course']}) ")
        print(f"  - Top Career Rec: {top_career_rec} (Correct: {item['correct_career']}) ")

        if top_course_rec == item['correct_course']:
            correct_course_predictions += 1
        
        if top_career_rec == item['correct_career']:
            correct_career_predictions += 1

    # --- 4. Calculate and Print Metrics ---
    total_items = len(ground_truth_data)
    course_accuracy = (correct_course_predictions / total_items) * 100
    career_accuracy = (correct_career_predictions / total_items) * 100

    print("\n--- Evaluation Results ---")
    print(f"Based on the small sample dataset of {total_items} items:")
    print(f"Course Recommendation Accuracy (Precision@1): {course_accuracy:.2f}%")
    print(f"Career Recommendation Accuracy (Precision@1): {career_accuracy:.2f}%")
    print("\nNOTE: These metrics are for demonstration only. A large, validated ground truth dataset is required for meaningful evaluation.")


if __name__ == "__main__":
    evaluate_recommender()
