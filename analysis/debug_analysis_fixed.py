import json
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import datetime

print("=" * 60)
print("�� SDSS SkyServer Astronomical Classification - DEBUG VERSION")
print("=" * 60)

# Print current working directory
current_dir = os.getcwd()
print(f"\n📁 Current directory: {current_dir}")

# Create sample data
print("\n📊 Creating sample dataset...")
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
    'redshift': np.concatenate([
        np.random.exponential(0.1, 400),
        np.random.exponential(0.3, 500),
        np.random.exponential(1.0, 100)
    ])
}

classes = []
for i in range(n_samples):
    if data['redshift'][i] < 0.15:
        classes.append('STAR')
    elif data['redshift'][i] < 0.4:
        classes.append('GALAXY')
    else:
        classes.append('QSO')

data['class'] = classes
df = pd.DataFrame(data)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"✅ Dataset created")

# Prepare features and target
feature_columns = ['ra', 'dec', 'u', 'g', 'r', 'i', 'z', 'redshift']
X = df[feature_columns]
y = df['class']

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
print("🌲 Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)

print(f"✅ Model trained. Test accuracy: {test_acc:.4f}")

# Prepare results - FIXED: feature_importances_ (correct spelling)
results = {
    'project_info': {
        'name': 'SDSS SkyServer Astronomical Classification',
        'dataset_size': len(df),
        'features': feature_columns,
        'classes': list(le.classes_),
        'class_counts': df['class'].value_counts().to_dict()
    },
    'performance': {
        'testing_accuracy': float(test_acc)
    },
    'feature_importance': dict(zip(feature_columns, map(float, model.feature_importances_))),
    'generated_date': datetime.datetime.now().isoformat()
}

# Try multiple save locations
save_locations = [
    '../results/results.json',
    './results.json',
    '/Users/nivetham/Documents/space-astronomy-ml/results/results.json',
    '../website/results.json'
]

print("\n💾 Attempting to save results to multiple locations...")

for location in save_locations:
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(location), exist_ok=True)
        
        with open(location, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Verify file was created
        if os.path.exists(location):
            file_size = os.path.getsize(location)
            print(f"✅ Saved to: {location} ({file_size} bytes)")
        else:
            print(f"❌ Failed to save to: {location}")
    except Exception as e:
        print(f"❌ Error saving to {location}: {e}")

print("\n📂 Current directory contents:")
os.system('ls -la | grep results.json || echo "No results.json in current directory"')

print("\n📂 Results directory contents:")
os.system('ls -la ../results/ 2>/dev/null || echo "Results directory not found"')

print("\n📂 Website directory contents:")
os.system('ls -la ../website/ 2>/dev/null | grep results.json || echo "No results.json in website directory"')

print("\n✅ Debug complete!")
