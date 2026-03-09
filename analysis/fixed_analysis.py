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
print("🚀 SDSS SkyServer Astronomical Classification")
print("=" * 60)

# Create sample data directly (no CSV needed)
print("\n📊 Creating sample dataset...")
np.random.seed(42)
n_samples = 1000

# Generate features that mimic SDSS data
data = {
    'ra': np.random.uniform(0, 360, n_samples),
    'dec': np.random.uniform(-90, 90, n_samples),
    'u': np.random.uniform(12, 22, n_samples),
    'g': np.random.uniform(12, 22, n_samples),
    'r': np.random.uniform(12, 22, n_samples),
    'i': np.random.uniform(12, 22, n_samples),
    'z': np.random.uniform(12, 22, n_samples),
    'redshift': np.concatenate([
        np.random.exponential(0.1, 400),  # Stars (low redshift)
        np.random.exponential(0.3, 500),  # Galaxies (medium redshift)
        np.random.exponential(1.0, 100)   # Quasars (high redshift)
    ])
}

# Create target classes based on redshift
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

# Shuffle the data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"✅ Dataset created: {df.shape[0]} observations, {df.shape[1]} features")

# Display class distribution
print("\n📊 Class Distribution:")
class_counts = df['class'].value_counts()
for cls, count in class_counts.items():
    print(f"   {cls}: {count} ({count/len(df)*100:.1f}%)")

# Prepare features and target
feature_columns = ['ra', 'dec', 'u', 'g', 'r', 'i', 'z', 'redshift']
X = df[feature_columns]
y = df['class']

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(f"\n🔤 Class mapping: {class_mapping}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\n📈 Train-Test Split:")
print(f"   Training set: {X_train.shape[0]} samples")
print(f"   Testing set: {X_test.shape[0]} samples")

# Train Random Forest model
print("\n🌲 Training Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracies
train_pred = model.predict(X_train)
train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, y_pred)

print(f"\n📊 Model Performance:")
print(f"   Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"   Testing Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")

# Feature importance
importance = model.feature_importances_
feature_importance = {}
for feature, imp in zip(feature_columns, importance):
    feature_importance[feature] = float(imp)

# Sort features by importance
sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
print("\n🔍 Feature Importance:")
for feature, imp in sorted_features:
    print(f"   {feature}: {imp:.4f}")

# Create sample predictions (using integer indexing for numpy arrays)
sample_predictions = []
for i in range(min(10, len(X_test))):
    # Get prediction probability
    X_sample = X_test.iloc[i:i+1] if hasattr(X_test, 'iloc') else X_test[i:i+1].reshape(1, -1)
    proba = model.predict_proba(X_sample)[0]
    confidence = float(max(proba))
    
    sample_predictions.append({
        'actual': le.inverse_transform([y_test[i]])[0],
        'predicted': le.inverse_transform([y_pred[i]])[0],
        'confidence': confidence,
        'redshift': float(X_test.iloc[i]['redshift'] if hasattr(X_test, 'iloc') else X_test[i][-1])
    })

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
    'sample_predictions': sample_predictions,
    'generated_date': datetime.datetime.now().isoformat()
}

# Save to JSON
output_path = '../results/results.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved to: {output_path}")
print("=" * 60)

# Verify the file was created
if os.path.exists(output_path):
    print(f"✅ File exists: {output_path}")
    file_size = os.path.getsize(output_path)
    print(f"📁 File size: {file_size} bytes")
else:
    print(f"❌ File not found: {output_path}")
