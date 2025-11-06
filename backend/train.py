import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from app.custom_transformers import MLBWrapper
import joblib
import json

print("Starting model training process...")

# --- 1. Load Data ---
data_path = './data/synthetic_student_data.csv'
df = pd.read_csv(data_path)

# --- 2. Feature Engineering ---

# Define grade to points mapping
grade_points = {
    'A': 12, 'A-': 11, 'B+': 10, 'B': 9, 'B-': 8,
    'C+': 7, 'C': 6, 'C-': 5, 'D+': 4, 'D': 3, 'D-': 2, 'E': 1
}

grade_columns = ['Mathematics', 'Kiswahili', 'English', 'Arabic', 'German', 'French', 'Chemistry', 'Physics', 'Biology', 'Home Science', 'Agriculture', 'Computer Studies', 'History', 'Geography', 'Religious Education', 'Life Skills', 'Business Studies', 'Music', 'Art and Design', 'Drawing and Design', 'Building Construction', 'Power and Mechanics', 'Metalwork', 'Aviation', 'Woodwork', 'Electronics']

# Convert grades to points
for col in grade_columns:
    df[col] = df[col].map(grade_points)

# Handle interests and skills (convert comma-separated strings to lists)
def split_str(s):
    return [item.strip() for item in s.split(',')] if isinstance(s, str) else []

df['interests'] = df['interests'].apply(split_str)
df['skills'] = df['skills'].apply(split_str)

# --- 3. Prepare for Training ---

# Define features (X) and target (y)
X = df[grade_columns + ['interests', 'skills']]
y = df['target_course_id']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 4. Create a Preprocessing and Modeling Pipeline ---



# Define the column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('grades', 'passthrough', grade_columns),
        ('interests_bin', MLBWrapper(), 'interests'),
        ('skills_bin', MLBWrapper(), 'skills')
    ],
    remainder='drop'
)

# Define the full pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
])

# --- 5. Train the Model ---
print("Training the Random Forest model...")
model_pipeline.fit(X_train, y_train)

# --- 6. Evaluate the Model ---
accuracy = model_pipeline.score(X_test, y_test)
print(f"Model training complete. Accuracy on test set: {accuracy:.2f}")

# --- 7. Save Metrics and Model ---

# Save metrics
metrics_path = './app/model_metrics.json'
with open(metrics_path, 'w') as f:
    json.dump({"course_model_accuracy": accuracy}, f)
print(f"Model metrics saved to {metrics_path}")

# Save the Model
model_path = './app/course_recommender_model.joblib'
joblib.dump(model_pipeline, model_path)
print(f"Model pipeline saved to {model_path}")
