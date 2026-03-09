import json
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import datetime

print("=" * 60)
print("🚀 SDSS SkyServer Astronomical Classification")
print("=" * 60)

# Check if CSV exists
csv_file = "Skyserver_SQL2_27_2018 6_51_39 PM.csv"

if not os.path.exists(csv_file):
    print(f"\n❌ CSV file not found: {csv_file}")
    print("Creating sample data instead...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'ra': np.random.uniform(0, 360, n_samples),
        'dec': np.random.uniform(-90, 90, n_samples),
        'u': np.random.uniform(12, 22, n_samples),
        'g': np.random.uniform(12, 22, n_samples),
        'r': np.random.uniform(12, 22, n_samples),
        'i': np.random.uniform(12, 22, n_samples),
        'z': np.random.uniform(12, 22, n_samples),
        'redshift': np.random.exponential(0.5, n_samples),
    }
    
    # Create target classes
    classes = []
    for i in range(n_samples):
        if data['redshift'][i] < 0.1:
            classes.append('STAR')
        elif data['redshift'][i] < 0.5:
            classes.append('GALAXY')
        else:
            classes.append('QSO')
    
    data['class'] = classes
    df = pd.DataFrame(data)
    print("✅ Sample data created!")
else:
    print(f"\n📂 Loading dataset: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"✅ Dataset loaded: {df.shape[0]} observations, {df.shape[1]} features")

# Display class distribution
print("\n📊 Class Distribution:")
class_counts = df['class'].value_counts()
for cls, count in class_counts.items():
    print(f"   {cls}: {count} ({count/len(df)*100:.1f}%)")

# Prepare features and target
feature_columns = [col for col in df.columns if col != 'class']
X = df[feature_columns]
y = df['class']

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Train Random Forest model
print("\n🌲 Training Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracies
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, y_pred)

print(f"\n📊 Model Performance:")
print(f"   Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"   Testing Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")

# Feature importance
importance = model.feature_importances_
feature_importance = dict(zip(feature_columns, importance))

# Create results directory
os.makedirs('../results', exist_ok=True)

# Prepare results for JSON
results = {
    'project_info': {
        'name': 'SDSS SkyServer Astronomical Classification',
        'description': 'Classification of celestial objects into Stars, Galaxies, and Quasars using Random Forest',
        'dataset_size': len(df),
        'features': feature_columns,
        'classes': list(le.classes_),
        'class_counts': class_counts.to_dict()
    },
    'model_info': {
        'algorithm': 'Random Forest Classifier',
        'n_estimators': 100,
        'random_state': 42
    },
    'performance': {
        'training_accuracy': float(train_acc),
        'testing_accuracy': float(test_acc)
    },
    'feature_importance': feature_importance,
    'sample_predictions': [
        {
            'actual': le.inverse_transform([y_test.iloc[i]])[0],
            'predicted': le.inverse_transform([y_pred[i]])[0],
            'confidence': float(max(model.predict_proba(X_test.iloc[i:i+1])[0]))
        }
        for i in range(min(5, len(X_test)))
    ],
    'generated_date': datetime.datetime.now().isoformat()
}

# Save to JSON
output_path = '../results/results.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved to: {output_path}")
print("=" * 60)
